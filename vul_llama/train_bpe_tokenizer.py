import os
import subprocess
from pathlib import Path
from transformers import AutoTokenizer

import logging

def setup_logger(log_path):
    """設定 logger，將輸出寫入指定 txt 檔案"""
    logger = logging.getLogger("StatLogger")
    logger.setLevel(logging.INFO)

    # 清空舊 handler（避免重複寫入）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 檔案輸出
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    # 終端輸出
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

def clone_aosp_modules(aosp_dir, logger):
    """下載 AOSP 中常用模組的 Java 原始碼"""
    os.makedirs(aosp_dir, exist_ok=True)
    repos = [
        "https://android.googlesource.com/platform/frameworks/base",
    ]

    for repo in repos:
        repo_name = repo.split("/")[-1]
        dest_path = os.path.join(aosp_dir, repo_name)
        if not os.path.exists(dest_path):
            logger.info(f"📥 Cloning {repo} ...")
            subprocess.run(["git", "clone", "--depth", "1", repo, dest_path])
        else:
            logger.info(f"✅ {dest_path} already exists, skipping.")
    
    return dest_path

def extract_codellama_tokenizer_vocab(tokenizer_name: str, output_path: str):
    """從原始 CodeLLaMA tokenizer 抽出 vocab token 並寫成文字檔模擬語料"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_tokens = list(tokenizer.get_vocab().keys())

    with open(output_path, "w", encoding="utf-8") as f:
        for token in vocab_tokens:
            if token.startswith("▁"):
                f.write(token.replace("▁", "") + "\n")

    print(f"✅ 原始 vocab token 語料已儲存至 {output_path}")


def collect_java_kotlin_files(root_dir, extensions=(".java", ".kt")):
    """收集指定副檔名的程式碼檔案"""
    files = []
    for path in Path(root_dir).rglob("*"):
        if path.suffix in extensions:
            files.append(str(path))
    return files


def build_aosp_corpus(file_list, output_path: str, logger=None):
    """將 AOSP 程式碼整理成訓練語料（包含換行與縮排）"""
    with open(output_path, "w", encoding="utf-8") as out_f:
        for file_path in file_list:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    out_f.write(f.read())
                    out_f.write("\n\n")
            except Exception as e:
                logger.warning(f"[Warning] Failed to read {file_path}: {e}")
    logger.info(f"✅ AOSP corpus 儲存至 {output_path}")


def merge_corpora(vocab_path: str, aosp_path: str, output_path: str, logger):
    """合併 vocab 語料與 AOSP 語料"""
    with open(output_path, "w", encoding="utf-8") as out_f:
        for path in [vocab_path, aosp_path]:
            with open(path, "r", encoding="utf-8") as f:
                out_f.write(f.read())
                out_f.write("\n")
    logger.info(f"✅ 合併語料儲存至 {output_path}")


def train_tokenizer_from_corpus(corpus_path: str, base_tokenizer: str, save_dir: str, vocab_size: int = 32000, logger=None):
    """用合併後語料重新訓練 tokenizer，保留 CodeLLaMA 的行為"""
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer, use_fase=True)

    def get_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    new_tokenizer = tokenizer.train_new_from_iterator(
        get_lines(corpus_path),
        vocab_size=vocab_size
    )
    new_tokenizer.save_pretrained(save_dir)
    logger.info(f"✅ 新 tokenizer 已儲存至 {save_dir}")

import os
import sentencepiece as spm
from transformers import LlamaTokenizer

def train_sentencepiece_tokenizer(corpus_path, save_dir, vocab_size=32000, logger=None):
    """使用 SentencePiece 訓練 tokenizer，保留 byte fallback 與 Llama 相容"""
    os.makedirs(save_dir, exist_ok=True)
    model_prefix = os.path.join(save_dir, "spm")

    logger.info("🚀 開始訓練 SentencePiece tokenizer ...")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,            # ✅ 保留 <0x0A> 等特殊符號
        train_extremely_large_corpus=True,
        pad_id=0, pad_piece="[PAD]",
        unk_id=1, unk_piece="[UNK]",
        bos_id=2, bos_piece="[BOS]",
        eos_id=3, eos_piece="[EOS]"
    )
    logger.info(f"✅ SentencePiece 模型儲存於：{model_prefix}.model")

    logger.info("📦 匯出為 HuggingFace 格式 tokenizer")
    tokenizer = LlamaTokenizer(vocab_file=f"{model_prefix}.model")
    tokenizer.save_pretrained(save_dir)
    logger.info(f"✅ HuggingFace tokenizer 儲存至：{save_dir}/")



def test(tokenizer_path: str, logger):
    """測試新訓練好的 tokenizer 分詞行為，與原始 CodeLLaMA 分詞器對照"""

    logger.info("🔍 測試新 tokenizer 分詞效果（與原始 tokenizer 比較）：")

    test_cases = [
        # 一般 Java code
        "public static void main(String[] args) { System.out.println(\"Hello World\"); }",
        "Intent intent = new Intent(this, TargetActivity.class);",
        "if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) { requestPermissions(); }",

        # 4. 換行測試
        "String a = \"line1\";\nString b = \"line2\";",

        # 5. Tab 測試（注意：這是 tab，不是 4 個空格）
        "\tif (x == 1) {\n\t\tdoSomething();\n\t}",

        # 6. 多空白測試（四個 space）
        "    return a + b;",

        # 7. AOSP-style function
        "private void checkPermission() {\n    if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {\n        // ask again\n    }\n}",
    
        # 其他 Android 常量測試
        # 8. Android permission 常量
        "if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) { return; }",

        # 9. Android settings
        "Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION);",

        # 10. TelephonyManager 方法
        "String deviceId = telephonyManager.getDeviceId();",

        # 11. SharedPreferences 編輯器
        "SharedPreferences.Editor editor = prefs.edit(); editor.putString(\"key\", \"value\").apply();",

        # 12. 取得資源顏色
        "int white = ContextCompat.getColor(context, R.color.white);",

        # 13. NotificationChannel 建立
        "NotificationChannel channel = new NotificationChannel(CHANNEL_ID, \"Channel name\", NotificationManager.IMPORTANCE_HIGH);",

        # 14. 使用 ContentResolver 查詢聯絡人
        "Cursor cursor = getContentResolver().query(ContactsContract.Contacts.CONTENT_URI, null, null, null, null);",
    ]

    original_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", use_fast=True)
    new_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    
    vocab = new_tokenizer.get_vocab()
    # 檢查是否包含常見 fallback byte
    for b in ["<0x0A>", "<0x09>", "<0x20>"]:
        print(f"{b} in vocab? {'Yes' if b in vocab else 'No'}")


    for idx, code in enumerate(test_cases):
        logger.info(f"\n【Test Case {idx+1}】原始碼：\n{code}\n")

        orig_tokens = original_tokenizer.tokenize(code)
        new_tokens = new_tokenizer.tokenize(code)

        logger.info(f"🧠 原始 tokenizer 分詞：\n{orig_tokens}\n")
        logger.info(f"🆕 新 tokenizer 分詞：\n{new_tokens}\n")

        if orig_tokens == new_tokens:
            logger.info("✅ 分詞完全一致\n")
        else:
            logger.info("❗ 分詞結果不同\n")



def main():
    log_path = "./vul_llama/train_logs/train_bpe_tokenizer.log"

    logger = setup_logger(log_path)

    BASE_TOKENIZER = "codellama/CodeLlama-7b-hf"
    AOSP_ROOT = "./vul_llama/tokenizer/AOSP"
    AOSP_DIR = clone_aosp_modules(AOSP_ROOT, logger)
    SAVE_DIR = "./vul_llama/android_tokenizer"

    VOCAB_CORPUS = "./vul_llama/tokenizer/codellama_vocab.txt"
    AOSP_CORPUS = "./vul_llama/tokenizer/aosp_corpus.txt"
    MERGED_CORPUS = "./vul_llama/tokenizer/merged_corpus.txt"

    extract_codellama_tokenizer_vocab(BASE_TOKENIZER, VOCAB_CORPUS)    
    code_files = collect_java_kotlin_files(AOSP_DIR)
    logger.info(f"共找到 {len(code_files)} 個 AOSP 檔案")
    build_aosp_corpus(code_files, AOSP_CORPUS, logger=logger)
    merge_corpora(VOCAB_CORPUS, AOSP_CORPUS, MERGED_CORPUS, logger=logger)
    train_tokenizer_from_corpus(MERGED_CORPUS, BASE_TOKENIZER, SAVE_DIR, vocab_size=32000, logger=logger)
    SAVE_DIR = "./vul_llama/android_tokenizer_spm"
    train_sentencepiece_tokenizer(MERGED_CORPUS, SAVE_DIR, vocab_size=32000, logger=logger)

    test(SAVE_DIR, logger)


if __name__ == "__main__":
    main()