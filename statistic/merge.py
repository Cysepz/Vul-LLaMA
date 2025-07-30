import pandas as pd

def clean_cwe_id(val):
    """將 list 形式的 cwe_id（不論是字串還是 list）統一轉為 'CWE-xxx' 字串"""
    if pd.isna(val):
        return ""
    if isinstance(val, list):
        return val[0] if val else ""
    if isinstance(val, str):
        # 嘗試移除多餘符號，只取第一個項目
        match = re.findall(r'CWE-\d+', val)
        return match[0] if match else val
    return str(val)

def merge_vl_into_test(test_csv, vl_csv, output_csv):
    # 讀取 test 和 VL 的資料
    df_test = pd.read_csv(test_csv)
    df_vl = pd.read_csv(vl_csv)

    # 重命名 VL 中要合併進來的欄位
    df_vl = df_vl.rename(columns={
        "function_pred": "VL_func_pred",
        "rank": "VL_rank"
    })

    # 指定比對欄位
    key_columns = ["apk_name", "file", "fullName", "merge_cwe_id", "function_name", "line_code"]

    df_test["merge_cwe_id"] = df_test["cwe_id"].apply(clean_cwe_id)
    df_vl["merge_cwe_id"] = df_vl["cwe_id"].apply(clean_cwe_id)

    # 加 row_id 保順序
    df_test["row_id"] = range(len(df_test))

    # 進行 merge（左表 test 為主）
    merged = pd.merge(df_test, df_vl[key_columns + ["VL_func_pred", "VL_rank"]],
                         on=key_columns, how="left")

    # 排序並每筆 test 選出最小 rank / 有預測的那筆
    merged = merged.sort_values(by=["row_id", "VL_func_pred", "VL_rank"], ascending=[True, False, True])
    merged_dedup = merged.drop_duplicates(subset=["row_id"], keep="first")

    merged_dedup = merged_dedup.sort_values("row_id").drop(columns=["row_id", "merge_cwe_id"])
    merged_dedup.to_csv(output_csv, index=False)
    print(f"✅ VL 智慧合併完成，輸出至 {output_csv}")


import re

def normalize_linecode(code: str) -> str:
    """將 linecode 正規化，去除空白、縮排、換行等，模仿 LineVul Tokenizer 行為"""
    if pd.isna(code):
        return ""
    
    # 移除所有空白（含縮排、tab、換行）
    code = re.sub(r"\s+", "", str(code))

    # 可選：將 "new AdInformationJsInterface" → "newAdInformationJsInterface"
    code = re.sub(r'new([A-Za-z_][A-Za-z0-9_]*)', lambda m: "new" + m.group(1), code)

    return code

def merge_linevul_with_normalized_linecode(test_csv, linevul_csv, output_csv, prefix="VL"):
    """
    合併 LineVul 結果（VL 或 LV）到 test.csv，針對 linecode 使用 normalize 對齊

    prefix: "VL" or "LV"，用來命名 function_pred → VL_func_pred
    """
    # 讀入 CSV
    df_test = pd.read_csv(test_csv)
    df_lv = pd.read_csv(linevul_csv)

    # 加入 row_id，追蹤原始順序
    df_test["row_id"] = range(len(df_test))

    key_cols = ["apk_name", "file", "fullName", "cwe_id", "function_name"]

    # 加入 normalized linecode 欄位
    df_test["norm_linecode"] = df_test["line_code"].apply(normalize_linecode)
    df_lv["norm_linecode"] = df_lv["line_code"].apply(normalize_linecode)

    # 重命名欄位以避免衝突
    df_lv = df_lv.rename(columns={
        "function_pred": f"{prefix}_func_pred",
        "rank": f"{prefix}_rank"
    })

    # # 用 key + normalized linecode 來合併
    # df_merged = pd.merge(df_test,
    #                      df_lv[key_cols + ["norm_linecode", f"{prefix}_func_pred", f"{prefix}_rank"]],
    #                      on=key_cols + ["norm_linecode"],
    #                      how="left")

    # # 丟掉中間使用的欄位
    # df_merged.drop(columns=["norm_linecode"], inplace=True)

    # df_merged.drop_duplicates(subset=["apk_name", "file", "fullName", "cwe_id", "function_name", "line_code"], inplace=True)


    # # 輸出
    # df_merged.to_csv(output_csv, index=False)
    # print(f"✅ {prefix} 合併完成，輸出至 {output_csv}")

    # 多對多 merge：會產生多筆匹配組合
    merged = pd.merge(df_test, df_lv[key_cols + ["norm_linecode", f"{prefix}_func_pred", f"{prefix}_rank"]],
                      on=key_cols + ["norm_linecode"], how="left")

    # 優先保留 rank 最小或 func_pred=True 的那筆
    merged = merged.sort_values(by=["row_id", f"{prefix}_func_pred", f"{prefix}_rank"], ascending=[True, False, True])
    merged_dedup = merged.drop_duplicates(subset=["row_id"], keep="first")

    # 回復為原本欄位順序
    merged_dedup = merged_dedup.sort_values(by="row_id")
    merged_dedup = merged_dedup.drop(columns=["row_id", "norm_linecode"])

    # 儲存
    merged_dedup.to_csv(output_csv, index=False)
    print(f"✅ LineVul 智慧合併完成，輸出至 {output_csv}")

def merge_linevul_smart(test_csv, linevul_csv, output_csv, prefix="LV"):
    """
    智慧合併：確保與 test.csv 筆數一致，並為每筆 test 資料選出最佳匹配預測
    """
    df_test = pd.read_csv(test_csv)
    df_lv = pd.read_csv(linevul_csv)

    # 加入 row_id，追蹤原始順序
    df_test["row_id"] = range(len(df_test))

    key_cols = ["apk_name", "file", "function_name"]

    # 正規化 line_code
    df_test["norm_linecode"] = df_test["line_code"].apply(normalize_linecode)
    df_lv["norm_linecode"] = df_lv["line_code"].apply(normalize_linecode)

    # 重命名 linevul 欄位
    df_lv = df_lv.rename(columns={
        "function_pred": f"{prefix}_func_pred",
        "rank": f"{prefix}_rank"
    })

    # 多對多 merge：會產生多筆匹配組合
    merged = pd.merge(df_test, df_lv[key_cols + ["norm_linecode", f"{prefix}_func_pred", f"{prefix}_rank"]],
                      on=key_cols + ["norm_linecode"], how="left")

    # 優先保留 rank 最小或 func_pred=True 的那筆
    merged = merged.sort_values(by=["row_id", f"{prefix}_func_pred", f"{prefix}_rank"], ascending=[True, False, True])
    merged_dedup = merged.drop_duplicates(subset=["row_id"], keep="first")

    # 回復為原本欄位順序
    merged_dedup = merged_dedup.sort_values(by="row_id")
    merged_dedup = merged_dedup.drop(columns=["row_id", "norm_linecode"])

    # 儲存
    merged_dedup.to_csv(output_csv, index=False)
    print(f"✅ 智慧合併完成（筆數一致：{len(df_test)} 筆），輸出至 {output_csv}")


def main():
    ground_truth_csv = "/home/peggy/VulLlama/statistic/test.csv"
    vul_llama_result_csv = "/home/peggy/VulLlama/statistic/0629_vulllama_b4_pred_result.csv"
    line_vul_result_csv = "/home/peggy/VulLlama/statistic/0629_linevul_pred_results.csv"
    mid_output_csv = "/home/peggy/VulLlama/statistic/merge_mid.csv"
    final_output_csv = "/home/peggy/VulLlama/statistic/merge_final.csv"

    df = pd.read_csv(ground_truth_csv)
    print(f"讀取 test.csv，總筆數: {len(df)}")

    merge_vl_into_test(ground_truth_csv, vul_llama_result_csv, mid_output_csv)
    df = pd.read_csv(mid_output_csv)
    print(f"讀取 mid_output.csv，總筆數: {len(df)}")

    merge_linevul_with_normalized_linecode(mid_output_csv, line_vul_result_csv, final_output_csv, prefix="LV")
    df = pd.read_csv(final_output_csv)
    print(f"讀取 mid_output.csv，總筆數: {len(df)}")




if __name__ == "__main__":
    main()