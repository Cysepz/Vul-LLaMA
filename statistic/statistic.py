import pandas as pd
import ast
import logging

def setup_logger(output_txt_path):
    """設定 logger，將輸出寫入指定 txt 檔案"""
    logger = logging.getLogger("StatLogger")
    logger.setLevel(logging.INFO)

    # 清空舊 handler（避免重複寫入）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 檔案輸出
    file_handler = logging.FileHandler(output_txt_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    # 終端輸出
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

def stat_func_label_counts(df, logger):
    """統計 function_label 欄位為 0 和 1 的筆數"""
    count_0 = (df["target"] == 0).sum()
    count_1 = (df["target"] == 1).sum()
    total = len(df)

    logger.info("=== function_label 統計結果 ===")
    logger.info(f"總筆數: {total}")
    logger.info(f"function_label = 1 的筆數: {count_1}")
    logger.info(f"function_label = 0 的筆數: {count_0}")

def count_code_and_flaw_lines(csv_path, logger):
    df = pd.read_csv(csv_path)

    total_code_lines = 0
    total_flaw_lines = 0

    for idx, row in df.iterrows():
        # 統計 processed_func 行數
        func_code = row.get("processed_func", "")
        code_lines = func_code.strip().splitlines()
        total_code_lines += len(code_lines)

        # 統計 flaw 行數（line_labels 中為 1 的數量）
        line_labels_str = row.get("line_labels", "[]")
        try:
            line_labels = ast.literal_eval(line_labels_str)
            flaw_count = sum(1 for label in line_labels if label == 1)
            total_flaw_lines += flaw_count
        except Exception as e:
            logger.warning(f"⚠️ 第 {idx} 筆資料解析 line_labels 時出錯: {e}")

    logger.info("=== line_label 統計結果 ===")
    logger.info(f"總程式碼行數：{total_code_lines} 行")
    logger.info(f"漏洞行（line_label=1）總行數：{total_flaw_lines} 行")
    logger.info(f"漏洞行佔總程式碼行數比例：{(total_flaw_lines / total_code_lines) * 100:.2f}%")


def stat_flaw_line_counts(df, logger):
    """統計 target == 1 的資料中 flaw_line_index 的長度分布"""
    df_pos = df[df["target"] == 1]

    count_stats = {
        "1個值": 0,
        "2個值": 0,
        "3個值": 0,
        "4個值": 0,
        "5個值": 0,
        "6~10個值": 0,
        "大於10個值": 0
    }

    for _, row in df_pos.iterrows():
        raw = str(row.get("flaw_line_index", "")).strip()

        if raw == "" or pd.isna(raw):
            continue

        values = [x.strip() for x in raw.split(",") if x.strip() != ""]
        n = len(values)

        if n == 1:
            count_stats["1個值"] += 1
        elif n == 2:
            count_stats["2個值"] += 1
        elif n == 3:
            count_stats["3個值"] += 1
        elif n == 4:
            count_stats["4個值"] += 1
        elif n == 5:
            count_stats["5個值"] += 1
        elif n > 5 and n <= 10:
            count_stats["6~10個值"] += 1
        elif n > 10:
            count_stats["大於10個值"] += 1

    logger.info("=== 漏洞 function 中的漏洞行數統計（target == 1）===")
    for k, v in count_stats.items():
        logger.info(f"{k}: {v} 筆, 占總數 {v / len(df_pos) * 100:.2f}%")
    logger.info("")


# ---------- merge_result 分析 ----------
def normalize_lv_pred(val):
    if val in [True, "True", 1, "1"]:
        return 1
    elif val in [False, "False", 0, "0"]:
        return 0
    else:
        return pd.NA

def range_bucket(val):
    if 1 <= val <= 10:
        return str(int(val))  # 回傳 "1" ~ "10"
    elif val > 10:
        return "11+"
    else:
        return None


def analyze_merge_result(merge_csv_path, logger):
    df = pd.read_csv(merge_csv_path)

    df["LV_func_pred_bin"] = df["LV_func_pred"].apply(normalize_lv_pred)

    # 1. VL_func_pred != function_label
    vl_diff = (df["VL_func_pred"] != df["function_label"]).sum()

    # 2. LV_func_pred != function_label
    lv_diff = (df["LV_func_pred_bin"] != df["function_label"]).sum()

    # 3. Group LV_true but other rows missing pred or rank
    group_cols = ["apk_name", "file", "function_name", "cwe_id"]
    df["has_lv_pred"] = df["LV_func_pred"].notna()
    df["has_lv_rank"] = df["LV_rank"].notna()
    grouped = df.groupby(group_cols)
    incomplete_groups = 0
    for _, group in grouped:
        if (group["LV_func_pred_bin"] == 1).any():
            if ((group["LV_func_pred"].isna()) | (group["LV_rank"].isna())).sum() > 0:
                incomplete_groups += 1

    # 4–6. Rank 差分析
    df_rank = df.dropna(subset=["VL_rank", "LV_rank"]).copy()
    df_rank["VL_rank"] = pd.to_numeric(df_rank["VL_rank"], errors="coerce")
    df_rank["LV_rank"] = pd.to_numeric(df_rank["LV_rank"], errors="coerce")
    df_rank = df_rank.dropna(subset=["VL_rank", "LV_rank"])
    df_rank["diff"] = df_rank["LV_rank"] - df_rank["VL_rank"]

    # 統計區間
    vl_lt_lv = df_rank[df_rank["diff"] > 0]
    vl_lt_lv_count = len(vl_lt_lv)
    vl_lt_lv_buckets = vl_lt_lv["diff"].apply(range_bucket).value_counts().to_dict()

    vl_gt_lv = df_rank[df_rank["diff"] < 0]
    vl_gt_lv_count = len(vl_gt_lv)
    vl_gt_lv_buckets = (-vl_gt_lv["diff"]).apply(range_bucket).value_counts().to_dict()

    vl_eq_lv = df_rank[df_rank["diff"] == 0]
    vl_eq_lv_count = len(vl_eq_lv)
    vl_eq_lv_buckets = vl_eq_lv["VL_rank"].apply(range_bucket).value_counts().to_dict()

    # ========== logger 輸出 ==========
    logger.info("=== Merge result 預測分析===")
    logger.info(f"1. VL_func_pred ≠ function_label: {vl_diff} 筆")
    logger.info(f"2. LV_func_pred ≠ function_label: {lv_diff} 筆")
    logger.info(f"3. LineVul 判斷為 vuln func 但缺乏行級別分數的筆數: {incomplete_groups}")
    logger.info(f"4. VL_rank < LV_rank: {vl_lt_lv_count} 筆")
    for bucket in sorted(vl_lt_lv_buckets, key=lambda x: int(x) if x.isdigit() else 99):
        logger.info(f"   差距 {bucket}: {vl_lt_lv_buckets[bucket]} 筆")

    logger.info(f"5. VL_rank > LV_rank: {vl_gt_lv_count} 筆")
    for bucket in sorted(vl_gt_lv_buckets, key=lambda x: int(x) if x.isdigit() else 99):
        logger.info(f"   差距 {bucket}: {vl_gt_lv_buckets[bucket]} 筆")

    logger.info(f"6. VL_rank == LV_rank: {vl_eq_lv_count} 筆")
    for bucket in sorted(vl_eq_lv_buckets, key=lambda x: int(x) if x.isdigit() else 99):
        logger.info(f"   Rank 位於 {bucket}: {vl_eq_lv_buckets[bucket]} 筆")
    logger.info("")

def main():
    train_csv_path = "/home/peggy/VulLlama/data_linevul/train.csv"
    eval_csv_path = "/home/peggy/VulLlama/data_linevul/eval.csv"
    test_csv_path = "/home/peggy/VulLlama/data_linevul/test.csv"
    merge_csv_path = "/home/peggy/VulLlama/statistic/merge_final.csv"
    output_txt_path = "./statistic/statistic.txt"

    logger = setup_logger(output_txt_path)

    train_df = pd.read_csv(train_csv_path)
    logger.info(f"讀取資料集: {train_csv_path}，總筆數: {len(train_df)}")
    stat_func_label_counts(train_df, logger)
    count_code_and_flaw_lines(train_csv_path, logger)
    stat_flaw_line_counts(train_df, logger)

    eval_df = pd.read_csv(eval_csv_path)
    logger.info(f"讀取資料集: {eval_csv_path}，總筆數: {len(eval_df)}")
    stat_func_label_counts(eval_df, logger)
    count_code_and_flaw_lines(eval_csv_path, logger)
    stat_flaw_line_counts(eval_df, logger)

    test_df = pd.read_csv(test_csv_path)
    logger.info(f"讀取資料集: {test_csv_path}，總筆數: {len(test_df)}")
    stat_func_label_counts(test_df, logger)
    count_code_and_flaw_lines(test_csv_path, logger)
    stat_flaw_line_counts(test_df, logger)

    df = pd.read_csv(merge_csv_path)
    logger.info(f"讀取合併結果: {merge_csv_path}，總漏洞行: {len(df)}")
    analyze_merge_result(merge_csv_path, logger)

if __name__ == "__main__":
    main()
