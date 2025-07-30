import ast
import pandas as pd

def extract_flawed_lines(csv_path):
    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        try:
            line_numbers = ast.literal_eval(row["line_numbers"])
            line_labels = ast.literal_eval(row["line_labels"])
            # 注意：某些 processed_func 可能是用 \n 表示的，需要保證是正確換行
            processed_func = row["processed_func"].replace("\\n", "\n")
            lines_of_code = processed_func.splitlines()
            lines_of_code = lines_of_code[:len(line_numbers)]  # ⬅️ 關鍵裁切
        except Exception as e:
            print(f"解析失敗: {e}")
            continue

        # 保險檢查：三者長度一致
        if len(line_numbers) != len(line_labels) or len(line_numbers) != len(lines_of_code):
            print(f"⚠️ 行數不一致：line_numbers={len(line_numbers)}, line_labels={len(line_labels)}, lines_of_code={len(lines_of_code)}")
            continue

        for line_num, label, code in zip(line_numbers, line_labels, lines_of_code):
            if label == 1:
                results.append({
                    "apk_name": row["apk_name"],
                    "file": row["file"],
                    "fullName": row["fullName"],
                    "cwe_id": row["cwe_id"],
                    "description": row["description"],
                    "function_name": row["function_name"],
                    "function_label": row["target"],
                    "line_number": line_num,
                    "line_code": code,
                })

    return pd.DataFrame(results)

def new_rank_df(input_path, output_path):
    # 讀取 CSV
    df = pd.read_csv(input_path)

    # 以 function 為單位排序並加上排名
    df["rank"] = df.groupby(["apk_name", "file", "function_name"])["line_score"] \
                .rank(method="dense", ascending=False).astype(int)

    # 儲存結果
    df.to_csv(output_path, index=False)

    print("已新增 rank 欄位並輸出到:", output_path)

def select_vul_line(input_path, output_path):
    # 讀取 CSV
    df = pd.read_csv(input_path)

    # 僅保留 function_label == 1 且 line_label == 1 的資料
    df = df[(df["function_label"] == 1) & (df["line_label"] == 1)].copy()

    # 儲存結果
    df.to_csv(output_path, index=False)

    print("已選擇 function_label == 1 且 line_label == 1 的行並輸出到:", output_path)

def merge_vulllama_into_test(test_csv, vul_llama_csv, output_csv):
    # 讀取資料
    df_test = pd.read_csv(test_csv)
    df_vul = pd.read_csv(vul_llama_csv)

    # 確保欄位是字串，避免比對錯誤
    key_cols = ["apk_name", "file", "fullName", "line_code"]
    for col in key_cols:
        df_test[col] = df_test[col].astype(str)
        df_vul[col] = df_vul[col].astype(str)

    # 建立合併 key
    df_test["merge_key"] = df_test[key_cols].agg("|||".join, axis=1)
    df_vul["merge_key"] = df_vul[key_cols].agg("|||".join, axis=1)

    # 從 vul_llama 中擷取需要的欄位
    df_vul_subset = df_vul[["merge_key", "function_pred", "rank"]].copy()
    df_vul_subset.rename(columns={
        "function_pred": "function_pred_VL",
        "rank": "line_rank_VL"
    }, inplace=True)

    # 合併：以 test.csv 為主
    df_merged = df_test.merge(df_vul_subset, on="merge_key", how="left")

    # 清除中繼 key
    df_merged.drop(columns=["merge_key"], inplace=True)

    # 輸出
    df_merged.to_csv(output_csv, index=False)
    print(f"✅ 合併完成，結果輸出至: {output_csv}")

import re
def normalize_code(code: str) -> str:
    if pd.isna(code):
        return ""

    code = code.strip()

    # 1. 移除所有空白、tab、換行
    code = re.sub(r"\s+", "", code)

    # 2. 把類型 + 變數名稱的空白合併（已在 step 1 做完）

    # 3. 將 "new AdInformationJsInterface" → "newAdInformationJsInterface"
    code = re.sub(r'new([A-Za-z_][A-Za-z0-9_]*)', lambda m: "new" + m.group(1), code)

    return code


def merge_rank_df(vul_llama_path, linevul_path, merge_path):
    df_vul = pd.read_csv(vul_llama_path)
    df_linevul = pd.read_csv(linevul_path)

    if "rank" in df_vul.columns:
        df_vul.rename(columns={"rank": "rank_vulllama"}, inplace=True)

    key_cols = ["apk_name", "file", "fullName", "cwe_id"]

    for col in key_cols + ["line_code"]:
        df_vul[col] = df_vul[col].astype(str)
        df_linevul[col] = df_linevul[col].astype(str)

    df_vul["norm_line_code"] = df_vul["line_code"].apply(normalize_code)
    df_linevul["norm_line_code"] = df_linevul["line_code"].apply(normalize_code)

    df_vul["merge_key"] = df_vul[key_cols].agg("|||".join, axis=1) + "|||" + df_vul["norm_line_code"]
    df_linevul["merge_key"] = df_linevul[key_cols].agg("|||".join, axis=1) + "|||" + df_linevul["norm_line_code"]

    rank_map = df_linevul.set_index("merge_key")["rank"].to_dict()

    df_vul["rank_linevul"] = df_vul["merge_key"].map(rank_map)
    df_vul["rank_linevul"] = df_vul["rank_linevul"].apply(lambda x: str(int(x)) if pd.notnull(x) else "N/A")

    df_vul.drop(columns=["merge_key", "norm_line_code"], inplace=True)
    df_vul.to_csv(merge_path, index=False)
    print("✅ 強化鬆散比對完成，結果輸出至:", merge_path)


def main():
    test_path = "/home/peggy/VulLlama/data_linevul/test.csv"    # 原始測試集
    test_output = "/home/peggy/VulLlama/process_to_detail/test.csv" # 處理後的測試集
    vul_llama_path = "/home/peggy/VulLlama/vul_llama/results/0629_vulllama_b4_pred_result.csv"
    vul_llama_output = "/home/peggy/VulLlama/process_to_detail/0629_vulllama_b4_pred_result.csv"
    linevul_path = "/home/peggy/VulLlama/vul_llama/results/0629_linevul_pred_results.csv"
    linevul_output = "/home/peggy/VulLlama/process_to_detail/0629_linevul_pred_results.csv"
    merge_path = "/home/peggy/VulLlama/process_to_detail/0629_merge.csv"

    df_flaws = extract_flawed_lines(test_path)
    df_flaws.to_csv(test_output, index=False)
    print("✅ 已匯出包含 line_label == 1 的行")

    new_rank_df(vul_llama_path, vul_llama_output)
    new_rank_df(linevul_path, linevul_output)

    select_vul_line(vul_llama_output, vul_llama_output)
    select_vul_line(linevul_output, linevul_output)

    merge_vulllama_into_test(test_output, vul_llama_output, merge_path)

    # merge_rank_df(vul_llama_output, linevul_output, merge_path)

if __name__ == "__main__":
    main()