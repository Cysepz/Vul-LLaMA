# 以此為主
# 問題：pred_result.csv 還是怪怪的（會出現很多行的 attention score 都是 0） by 2025/6/2 20:38
# 問題：pred_result.csv 可以 show 出比較正常的東西了，但是需要回頭去跟 linevul 核對輸出的 attn_score 是否是正確的
import csv
import json
import pickle
import time
from typing import Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerFast 
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
import pandas as pd
from bitsandbytes.optim import Adam8bit
import os
import re

from cl_model import VulLlamaBaseline
from vl_model import VulLlamaModel, FunctionLevelClassifier, VulLlamaConfig

import logging
logger = logging.getLogger(__name__)

# ========= Dataset =========
def custom_collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for key in keys:
        out[key] = [item[key] for item in batch]
    return out

class CodeFunctionDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_length=512):
        self.data = json_data
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        code = ex["code"]
        label = ex["function_label"]
        tokenized = self.tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        if not code.strip():
            code = "int dummy() { return 0; }"

        return {
            "apk_name": ex.get("apk_name", ""),
            "cwe_id": ex.get("cwe_id", ""),
            "description": ex.get("description", ""),
            "fullName": ex.get("fullName", ""),
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label": torch.tensor(label),
            "code": code,
            "line_numbers": ex["line_numbers"],
            "line_labels": ex["line_labels"],
            "file": ex.get("file", f"example_{idx}"),
            "function_name": ex.get("function_name", f"func_{idx}")
        }

# ========= Train =========
def train(args, model, tokenizer, train_dataset, eval_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon) # baseline training version
    optimizer = Adam8bit(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-3)    # eps=1e-3 是解決 loss=Nan 的英雄!
    # schedular = torch.optim.lr_scheduler.OneCycleLR(

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss = 0.0
    best_f1 = 0.0
    model.zero_grad()

    for epoch in range(args.epochs):
        logger.info(f"=== Epoch {epoch+1}/{args.epochs} ===")
        with logging_redirect_tqdm():
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}")
            train_loss = 0.0
            tr_num = 0
            for step, batch in enumerate(bar):
                input_ids = torch.stack(batch["input_ids"]).to(args.device)
                attention_mask = torch.stack(batch["attention_mask"]).to(args.device)
                labels = torch.stack(batch["label"]).to(args.device)

                model.train()
                loss, _, _ = model(input_ids, attention_mask, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                train_loss += loss.item()
                tr_num += 1
                bar.set_postfix(loss=round(train_loss / tr_num, 4))
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % args.save_steps == 0:
                        results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)

                        if results['eval_f1'] > best_f1:
                            best_f1 = results['eval_f1']
                            logger.info("  "+"*"*20)
                            logger.info("  Best eval_f1: %.4f", best_f1)
                            logger.info("  "+"*"*20)

                            checkpoint_prefix = 'checkpoint-best-f1'
                            saved_model_dir = os.path.join(args.saved_model_dir, checkpoint_prefix)
                            os.makedirs(saved_model_dir, exist_ok=True)

                            model_to_save = model.module if hasattr(model, "module") else model
                            # output_model_path = os.path.join(saved_model_dir, f"{args.model_name}.bin")
                            # torch.save(model_to_save.state_dict(), output_model_path)
                            # logger.info("Saving best model checkpoint to %s", output_model_path)
                            # 決定儲存模型的路徑
                            checkpoint_path = os.path.join(saved_model_dir, f"{args.model_name}_{args.tokenizer}.pt")

                            # 儲存完整模型與訓練狀態
                            torch.save({
                                "epoch": epoch,  # 當前 epoch 數（或你用的迴圈變數）
                                "model_state_dict": model_to_save.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                # "scheduler_state_dict": scheduler.state_dict(),
                            }, checkpoint_path)

                            logger.info("Saved full checkpoint to %s", checkpoint_path)

        logger.info(f"Epoch {epoch+1} finished. Avg loss={round(train_loss / len(train_dataloader), 4)}")


# ========= Evaluation =========
def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not eval_when_training:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    model.to(args.device)

    eval_loss = 0.0
    nb_eval_steps = 0
    logits_list = []
    labels_list = []

    for batch in eval_dataloader:
        input_ids = torch.stack(batch["input_ids"]).to(args.device)
        attention_mask = torch.stack(batch["attention_mask"]).to(args.device)
        labels = torch.stack(batch["label"]).to(args.device)

        with torch.no_grad():
            loss, logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += loss.mean().item()
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

        nb_eval_steps += 1

    logits = torch.cat(logits_list, dim=0)
    y_trues = torch.cat(labels_list, dim=0)

    best_threshold = 0.5
    y_preds = (logits[:, 1] > best_threshold).int()
    
    accuracy = accuracy_score(y_trues.numpy(), y_preds.numpy())
    recall = recall_score(y_trues.numpy(), y_preds.numpy())
    precision = precision_score(y_trues.numpy(), y_preds.numpy())
    f1 = f1_score(y_trues.numpy(), y_preds.numpy())

    result = {
        "eval_accuracy": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    PrecisionRecallDisplay.from_predictions(y_trues, logits[:, 1], name=args.model_name)
    plt.savefig(f'eval_precision_recall_{args.model_name}.pdf')

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

# ========= Test =========
def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    logger.info("***** Running FULL TEST inference *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    model.to(args.device)

    output_result = []

    logits_list = []    # 模型對每一筆資料輸出的原始預測
    labels_list = []    # ground truth

    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = torch.stack(batch["input_ids"]).to(args.device)
        labels = torch.stack(batch["label"]).to(args.device)
        attention_mask = torch.stack(batch["attention_mask"]).to(args.device)

        with torch.no_grad():
            _, logits, attn = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())

    # === Function-level result ===
    # RQ1: function level 的表現
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0) 
    predictions = torch.argmax(logits, dim=1)

    vuln_probs = torch.sigmoid(logits[:, 1])  # 取 vulnerable 類別的 logit，使用 sigmoid 做正規化
    predictions = (vuln_probs > best_threshold).long()

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)

    logger.info("===== Function-Level Classification =====")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall   : {recall:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")

    # 製作 function level result dataframe
    # 1. 取得每筆 vulnerable 類別的預測分數（跟 LineVul 的 logits = [l[1] for l in logits] 一樣）
    vuln_probs_list = vuln_probs.detach().cpu().numpy().tolist()
    # 2. Ground truth 與預測結果
    y_trues = labels.detach().cpu().numpy().tolist()
    y_preds = predictions.detach().cpu().numpy().tolist()
    # 3. 產出 DataFrame：每一筆 function 的 [score, y_true, y_pred]
    func_result_df = generate_func_result_df(vuln_probs_list, y_trues, y_preds, args)
    # print(func_result_df)  # 印出前幾行確認結果
    # 4. 統計行數與 buggy 行數
    sum_lines, sum_flaw_lines = get_line_statistics(func_result_df)

    # Processing line level
    if args.do_sorting_by_line_scores:
        # (RQ3) Effort@TopK%Recall & Recall@TopK%LOC for the whole test set
        # flatten the logits
        per_func_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)   # 此處 batch size 一定要設 1，才會是以 function 為單位處理行級別注意力分數
        progress_bar = tqdm(per_func_dataloader, total=len(per_func_dataloader))
        
        all_pos_score_label = []
        all_neg_score_label = []
        index = 0
        total_pred_as_vul = 0

        start_time = time.time()
        for mini_batch in progress_bar:
            func_pred_score = func_result_df["logits"][index]
            func_code = func_result_df["code"][index]
            flaw_lines = func_result_df["buggy_lines"][index]  # 它就是一個 list，不用 values()

            # vulnerable function
            if func_pred_score > 0.5:
                total_pred_as_vul += 1
                all_lines_score_with_label, line_result = line_level_localization(
                    flaw_lines=flaw_lines,
                    tokenizer=tokenizer,
                    model=model,
                    mini_batch=mini_batch,
                    original_func=func_code,
                    args=args,
                    top_k_loc=[0.01, 0.05, 0.1, 0.2], # Top1LOC
                    top_k_constant=[1, 10, 20],  # Top10 Accuracy  
                    reasoning_method=args.reasoning_method,
                    index=index
                )
                all_pos_score_label.append(all_lines_score_with_label)
               
                for i, line_info in enumerate(line_result):
                    real_line_numbers = mini_batch["line_numbers"][0]  # 解包第一層
                    apk_name = mini_batch["apk_name"][0]
                    cwe_id = mini_batch["cwe_id"][0]
                    description = mini_batch["description"][0]
                    full_name = mini_batch["fullName"][0]
                    # print(mini_batch["line_numbers"][line_index][index]) 

                    output_result.append({
                        "index": index,
                        "apk_name": apk_name,
                        "file": func_result_df["file"][index],
                        "fullName": full_name,
                        "cwe_id": cwe_id,
                        "description": description,
                        "function_name": func_result_df["function_name"][index],
                        "function_label": y_trues[index],
                        "function_pred": y_preds[index],
                        "function_score": round(float(func_pred_score), 3),  # 小數第 3 位
                        "line_number": int(real_line_numbers[i]) if i < len(real_line_numbers) else i,
                        "line_code": line_info["line"],
                        "line_score": line_info["score"],
                        "line_label": int(line_info["label"])
                    })

            # non-vulnerable function
            else:
                all_lines_score_with_label, line_result = line_level_localization(
                    flaw_lines=flaw_lines,
                    tokenizer=tokenizer,
                    model=model,
                    mini_batch=mini_batch,
                    original_func=func_code,
                    args=args,
                    top_k_loc=[0.01, 0.05, 0.1, 0.2],  # Top1LOC
                    top_k_constant=[1, 10, 20],  # Top10 Accuracy
                    reasoning_method=args.reasoning_method,
                    index=index
                )
                all_neg_score_label.append(all_lines_score_with_label)
                for i, line_info in enumerate(line_result):
                    real_line_numbers = mini_batch["line_numbers"][0]  # 解包第一層
                    apk_name = mini_batch["apk_name"][0]
                    cwe_id = mini_batch["cwe_id"][0]
                    description = mini_batch["description"][0]
                    full_name = mini_batch["fullName"][0]
                    # print(mini_batch["line_numbers"][line_index][index]) 

                    output_result.append({
                        "index": index,
                        "apk_name": apk_name,
                        "file": func_result_df["file"][index],
                        "fullName": full_name,
                        "cwe_id": cwe_id,
                        "description": description,
                        "function_name": func_result_df["function_name"][index],
                        "function_label": y_trues[index],
                        "function_pred": y_preds[index],
                        "function_score": round(float(func_pred_score), 3),  # 小數第 3 位
                        "line_number": int(real_line_numbers[i]) if i < len(real_line_numbers) else i,
                        "line_code": line_info["line"],
                        "line_score": line_info["score"],
                        "line_label": int(line_info["label"])
                    })

            index += 1

        # 🕒 計時結束
        end_time = time.time()
        elapsed_time = end_time - start_time

        is_attention = True if args.reasoning_method == "attention" else False
        total_pos_lines, pos_rank_df  = rank_lines(all_pos_score_label, is_attention, ascending_ranking=False)
        
        # clean function 中的行評估
        if is_attention:
            # attention 分數低代表安全
            total_neg_lines, neg_rank_df = rank_lines(all_neg_score_label, is_attention, ascending_ranking=True)
        else:
            # logits 分數高代表安全（例如 MLP classifier）
            total_neg_lines, neg_rank_df = rank_lines(all_neg_score_label, is_attention, ascending_ranking=False)
        
        effort, inspected_line = top_k_effort(pos_rank_df, sum_lines, sum_flaw_lines, args.effort_at_n_percent_recall)

        recall_value = top_k_recall(pos_rank_df, neg_rank_df, sum_lines, sum_flaw_lines, args.recall_at_n_percent_loc)

        logger.info("===== Line-Level Cost Effective =====")
        logger.info(f"total functions predicted as vulnerable: {total_pred_as_vul}")

        logger.info(f"total predicted vulnerable lines: {total_pos_lines}")
        logger.info(f"total lines: {sum_lines}")
        logger.info(f"total flaw lines: {sum_flaw_lines}")
        
        vul_as_vul = sum(pos_rank_df["label"].tolist())
        logger.info(f"total flaw lines in predicted as vulnerable: {vul_as_vul}")
        logger.info(f"Effort at {args.effort_at_n_percent_recall} Recall: {effort}")
        logger.info(f"total inspected line to find out {args.effort_at_n_percent_recall} of flaw lines: {inspected_line}")
        logger.info(f"Recall at {args.recall_at_n_percent_loc} percent loc: {recall_value}")

        logger.info(f"Elapsed time for line-level localization: {elapsed_time:.2f} seconds")

        output_df = pd.DataFrame(output_result)
        output_csv_path = os.path.join(args.test_output_path)
        output_df.to_csv(output_csv_path, index=False, float_format='%.3f')  # 小數第 3 位
        logger.info(f"Function + Line-level results written to {output_csv_path}")

    if args.do_local_explanation:
        logger.info(f"===== Line-Level Location =====")
        # step 1: 取得 function-level TP
        correct_indices = [i for i in range(len(y_trues)) if y_trues[i] == y_preds[i]]
        tp_indices = [i for i in range(len(y_trues)) if y_trues[i] == y_preds[i] == 1]

        # step 2: 建立 dataloader（用 shuffle=False 以確保 index 對得上）
        per_function_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
        df = pd.read_json(args.test_data_file)
        
        # stats for line-level evaluation
        top_k_locs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        top_k_constant = [1, 2, 3, 5, 10, 20]   # Top-10 Accuracy
        sum_total_lines = 0
        total_flaw_lines = 0
        total_function = 0
        all_top_10_correct_idx = []
        all_top_10_not_correct_idx = []
        # for CodeBERT reasoning
        total_correctly_predicted_flaw_lines = [0 for _ in range(len(top_k_locs))]
        total_correctly_localized_function = [0 for _ in range(len(top_k_constant))]
        total_min_clean_lines_inspected = 0
        ifa_records = []
        top_10_acc_records = []
        total_max_clean_lines_inspected = 0
        # vulnerability exist but not applicable (flaw tokens are out of seq length)
        na_explanation_total = 0
        na_eval_results_512 = 0
        na_defective_data_point = 0
        # track progress
        progress_bar = tqdm(per_function_dataloader, total=len(per_function_dataloader))
        # used to locate the row in test data
        index = 0

        start_time = time.time()
        for mini_batch in progress_bar:
            # if true positive (vulnerable predicted as vulnerable), do explanation
            if index in tp_indices:
                # if flaw line exists
                # if not exist, the data is as type of float (nan)
                # 取出所有有漏洞的行號
                buggy_dict = df["buggy_lines"][index]

                if isinstance(buggy_dict, dict) and len(buggy_dict) > 0:
                    flaw_line_index = ";".join(buggy_dict.keys())  # 直接抓 dictionary 的 key（行號）
                    flaw_line = ";".join(buggy_dict.values())      # 直接抓 dictionary 的 value（原始碼內容）

                    if isinstance(flaw_line_index, str) and isinstance(flaw_line, str):
                        line_eval_results = \
                        line_level_localization_tp(flaw_lines=df["buggy_lines"][index],
                                                tokenizer=tokenizer, 
                                                model=model, 
                                                mini_batch=mini_batch, 
                                                original_func=df["code"][index], 
                                                args=args,
                                                top_k_loc=top_k_locs,
                                                top_k_constant=top_k_constant,
                                                reasoning_method=args.reasoning_method,
                                                index=index,
                                                write_invalid_data=False)
                        if line_eval_results == "NA":
                            na_explanation_total += 1 
                            na_eval_results_512 += 1
                        else:                       
                            total_function += 1
                            sum_total_lines += line_eval_results["total_lines"]
                            total_flaw_lines += line_eval_results["num_of_flaw_lines"]
                            # IFA metric
                            total_min_clean_lines_inspected += line_eval_results["min_clean_lines_inspected"]
                            
                            # For IFA Boxplot
                            ifa_records.append(line_eval_results["min_clean_lines_inspected"])
                            
                            # For Top-10 Acc Boxplot
                            # todo
                            #top_10_acc_records.append(line_eval_results[]) 
                            
                            # All effort metric
                            total_max_clean_lines_inspected += line_eval_results["max_clean_lines_inspected"]
                            for j in range(len(top_k_locs)):
                                total_correctly_predicted_flaw_lines[j] += line_eval_results["all_correctly_predicted_flaw_lines"][j]
                            # top 10 accuracy
                            for k in range(len(top_k_constant)):
                                total_correctly_localized_function[k] += line_eval_results["all_correctly_localized_function"][k]
                            # top 10 correct idx and not correct idx
                            if line_eval_results["top_10_correct_idx"] != []:
                                all_top_10_correct_idx.append(line_eval_results["top_10_correct_idx"][0])
                            if line_eval_results["top_10_not_correct_idx"] != []:
                                all_top_10_not_correct_idx.append(line_eval_results["top_10_not_correct_idx"][0]) 
                    else:
                        na_explanation_total += 1
                        na_defective_data_point += 1
            index += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        # write IFA records for IFA Boxplot
        with open(f"./vul_llama/results/ifa_records/ifa_{args.model_name}.txt", "w+") as f:
            f.write(str(ifa_records))
        # write Top-10 Acc records for Top-10 Accuracy Boxplot
        # todo
        #with open(f"./top_10_acc_records/top_10_acc_{reasoning_method}.txt", "w+") as f:
        #    f.write(str())

        logger.info(f"Total number of functions: {total_function}")
        logger.info(f"Total number of lines: {sum_total_lines}")
        logger.info(f"Total number of flaw lines: {total_flaw_lines}")
        logger.info(f"Total Explanation Not Applicable: {na_explanation_total}")
        logger.info(f"NA Eval Results (Out of 512 Tokens): {na_eval_results_512}")
        logger.info(f"NA Defective Data Point: {na_defective_data_point}")

        logger.info("\n")   # 換行
        logger.info(f"===== Line Level Result: Reasoning Method: {args.reasoning_method} =====")
        print_top_k_recall = [
            round(total_correctly_predicted_flaw_lines[i] / total_flaw_lines, 2) * 100 if total_flaw_lines > 0 else 0.0
            for i in range(len(top_k_locs))
        ]
        logger.info(f"Top-K LOC: {top_k_locs}")
        logger.info(f"Top-K Recall:{print_top_k_recall}")

        print_top_k_accuracy = [
            round(total_correctly_localized_function[i] / total_function, 2) * 100 if total_function > 0 else 0.0
            for i in range(len(top_k_constant))
        ]
        logger.info("Top-K Constant: %s", top_k_constant)
        logger.info(f"Top-K Accuracy: {print_top_k_accuracy}")

        logger.info(f"IFA:{round(total_min_clean_lines_inspected / total_function, 2) if total_function > 0 else 0.0}")

        auc_y = [
            round(total_correctly_predicted_flaw_lines[i] / total_flaw_lines, 2) if total_flaw_lines > 0 else 0.0
            for i in range(len(top_k_locs))
        ]
        logger.info(f"Recall@TopK%LOC AUC: {auc(x=top_k_locs, y=auc_y)}")
        
        logger.info(f"Total Effort: {round(total_max_clean_lines_inspected / sum_total_lines, 2) if sum_total_lines > 0 else 0.0}")
        logger.info(f"Avg Line in One Function: {int(sum_total_lines / total_function) if total_function > 0 else 0}")
        logger.info(f"Total Function: {total_function}")
        logger.info(f"All Top-10 Correct Index: {all_top_10_correct_idx}")
        logger.info(f"All Top-10 Not Correct Index: {all_top_10_not_correct_idx}")
                            
        with open('./vul_llama/results/line_level_correct_idx.pkl', 'wb') as f:
            pickle.dump(all_top_10_correct_idx, f)
        with open('./vul_llama/results/line_level_not_correct_idx.pkl', 'wb') as f:
            pickle.dump(all_top_10_not_correct_idx, f)

        logger.info(f"Elapsed time for line-level localization: {elapsed_time:.2f} seconds")

    # write results to CSV
    # df.to_csv(args.test_output_path, index=False)
    # logger.info(f"Line-level results written to {args.test_output_path}")

def generate_func_result_df(logits, y_trues, y_preds, args):
    # 讀取原始測試資料（JSON 格式）
    df = pd.read_json(args.test_data_file)

    # 使用 "code" 欄位計算每個 function 的行數
    all_codes = df["code"].tolist()
    all_num_lines = [len(code.strip().split("\n")) for code in all_codes]

    # 使用 "line_labels" 計算每個 function 中的 flaw line 數量
    all_line_labels = df["line_labels"].tolist()
    all_num_flaw_lines = [sum(labels) for labels in all_line_labels]

    # 檢查長度一致性
    assert len(logits) == len(y_trues) == len(y_preds) == len(all_num_flaw_lines)
    result_df = pd.DataFrame({
        "index": list(range(len(logits))),
        "file": df["file"],
        "function_name": df["function_name"],
        "code": df["code"],
        "num_lines": all_num_lines,
        "num_flaw_lines": all_num_flaw_lines,
        "logits": logits,
        "y_trues": y_trues,
        "y_preds": y_preds,
        "line_labels": df["line_labels"],
        "buggy_lines": df.get("buggy_lines", [{}] * len(df)),  # 若沒此欄位預設為空字典
        "function_label": df.get("function_label", [0] * len(df))  # 若沒此欄位預設為 0
    })

    return result_df

def get_line_statistics(result_df):
    total_lines = sum(result_df["num_lines"])
    total_flaw_lines = sum(result_df["num_flaw_lines"])
    return total_lines, total_flaw_lines

def rank_lines(all_lines_score_with_label, is_attention, ascending_ranking):
    # flatten the list
    all_lines_score_with_label = [line for lines in all_lines_score_with_label for line in lines]
    if is_attention:
        all_scores = [line[0].item() for line in all_lines_score_with_label]
    else:
        all_scores = [line[0] for line in all_lines_score_with_label]
    all_labels = [line[1] for line in all_lines_score_with_label]
    rank_df = pd.DataFrame({"score": all_scores, "label": all_labels})
    rank_df = rank_dataframe(rank_df, "score", ascending_ranking)
    return len(rank_df), rank_df


def rank_dataframe(df, rank_by: str, ascending: bool):
    df = df.sort_values(by=[rank_by], ascending=ascending)
    df = df.reset_index(drop=True)
    return df

def line_level_localization(flaw_lines, tokenizer, model, mini_batch, original_func, args, top_k_loc, top_k_constant, reasoning_method, index):
    flaw_line_seperator = "/~/"
    input_ids = mini_batch["input_ids"][0].unsqueeze(0).to(args.device)
    labels = mini_batch["label"][0].unsqueeze(0).to(args.device)
    attention_mask = mini_batch["attention_mask"][0].unsqueeze(0).to(args.device)
    input_ids = input_ids.to(args.device)

    ###########################################################################################################################
    # 1. 將 token id 轉為 token string（用於 line-level attention 對應）
    ids = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)

    # 2. 根據 tokenizer 類型清洗 token（支援 RoBERTa、CodeLLaMA、自訓 BPE）
    cleaned_tokens = []
    for token in all_tokens:
        if token is None:
            token = ""
        # RoBERTa only: Remove word boundary marker and handle newline
        if "roberta" in tokenizer.name_or_path.lower():
            token = token.replace("Ġ", "")  # Word-start marker in RoBERTa
            token = token.replace("ĉ", "Ċ")  # Newline symbol in RoBERTa
        cleaned_tokens.append(token)

    all_tokens = cleaned_tokens
    ###########################################################################################################################

    # 2. 將原始 function 的 code 拆為行
    original_lines = original_func.split("\n")

    # flaw line verification
    # get flaw tokens ground truth
    flaw_lines = get_all_flaw_lines(flaw_lines=flaw_lines, flaw_line_seperator=flaw_line_seperator)
    flaw_tokens_encoded = encode_all_lines(all_lines=flaw_lines, tokenizer=tokenizer)
    verified_flaw_lines = []
    for i in range(len(flaw_tokens_encoded)):
        encoded_flaw = ''.join(flaw_tokens_encoded[i])
        encoded_all = ''.join(all_tokens)
        if encoded_flaw in encoded_all:
            verified_flaw_lines.append(flaw_tokens_encoded[i])

    # Debug: print the verified flaw lines
    # logger.info(f"[Index {index}] 原始 flaw_lines:\n{flaw_lines}")
    # logger.info(f"[Index {index}] flaw_tokens_encoded:\n{flaw_tokens_encoded}")
    # logger.info(f"[Index {index}] all_tokens:\n{all_tokens}")
    # logger.info(f"[Index {index}] verified_flaw_lines:\n{verified_flaw_lines}")

    if reasoning_method == "attention":
        # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
        input_ids = input_ids.to(args.device)
        model.eval()
        model.to(args.device)
        with torch.no_grad():
            loss, logits, attentions = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        # take from tuple then take out mini-batch attention values

        # === 處理 attention shape ===
        if isinstance(attentions[0], torch.Tensor) and attentions[0].dim() == 4:
            attentions = attentions[0][0]  # → list of (seq_len, seq_len)
        elif isinstance(attentions[0], torch.Tensor) and attentions[0].dim() == 3:
            attentions = attentions[0] # → list of (seq_len, seq_len)
        attention = None

        # go into the layer
        for i in range(len(attentions)):
            layer_attention = attentions[i]
            # summerize the values of each token dot other tokens
            layer_attention = sum(layer_attention)
            if attention is None:
                attention = layer_attention
            else:
                attention += layer_attention
                
        # clean att score for <s> and </s>
        attention = clean_special_token_values(all_tokens, attention)
        # Debug: print each token with attention score
        # for tok, score in zip(all_tokens, attention):
        #     print(f"{tok:15s}: {score:.4f}")

        word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)
        # Debug: print each token with its word attention score
        # for tok, score in word_att_scores:
        #     try:
        #         print(f"***** {tok:15s}: {score:.4f}")
        #     except Exception as e:
        #         print(f"Error printing token={tok}, score={score}: {e}")

        all_lines_score, flaw_line_indices = get_all_lines_score(word_att_scores, verified_flaw_lines, args)

        # Debug: print each line with its score
        # 印出每一行的 attention 分數（由 token attention 加總而來）
        # print("=== All Line Scores ===")
        # for idx, score in enumerate(all_lines_score):
        #     print(f"Line {idx}: {score:.4f}")
        # 印出被標記為 flaw 的行 index（對應到 verified_flaw_lines）
        # print("\n=== Verified Flaw Line Indices ===")
        # print(flaw_line_indices)

        all_lines_score_with_label = \
        line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices, top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=False)
        original_lines = original_func.split("\n")
        
        # For print line results
        line_results = []
        for i, (score, label) in enumerate(all_lines_score_with_label):
            line_results.append({
                "line_number": i,
                "line": original_lines[i] if i < len(original_lines) else "",
                "score": float(score.item()) if isinstance(score, torch.Tensor) else float(score),
                "label": int(label)
            })

    return all_lines_score_with_label, line_results



def get_all_flaw_lines(flaw_lines: Union[str, dict], flaw_line_seperator: str = None) -> list:
    if isinstance(flaw_lines, dict):
        # 取出所有 value（即漏洞程式碼行），並 strip 掉前後空白
        return [line.strip() for line in flaw_lines.values()]
    elif isinstance(flaw_lines, str):
        flaw_lines = flaw_lines.strip(flaw_line_seperator)
        flaw_lines = flaw_lines.split(flaw_line_seperator)
        return [line.strip() for line in flaw_lines]
    else:
        return []

def encode_all_lines(all_lines: list[str], tokenizer) -> list[list[str]]:
    """
    將每行原始碼使用 tokenizer 編碼，回傳 token list 的 list。
    e.g., 輸入 3 行 code，輸出為 3 個 token list。
    """
    encoded = []
    for line in all_lines:
        encoded.append(encode_one_line(line=line, tokenizer=tokenizer))
    return encoded

def encode_one_line(line: str, tokenizer) -> list[str]:
    """
    將單行原始碼使用 tokenizer 編碼，並轉為 token 字串。
    """
    encoded_ids = tokenizer.encode(line, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
    # 移除特殊標記（視情況調整）
    tokens = [token.replace("Ġ", "").replace("ĉ", "Ċ") for token in tokens]
    return tokens

def clean_special_token_values(all_tokens, all_values):
    # all_tokens 是 tokenizer.convert_ids_to_tokens(input_ids)
    # all_values 是 attention.sum(dim=0) 的結果

    cleaned_values = all_values.clone()  # 不改原始
    for i, tok in enumerate(all_tokens):
        if tok in ['</s>', '<s>']:  # 視情況你也可以加 <pad>、'<unk>' 等
            cleaned_values[i] = 0
    return cleaned_values

def get_word_att_scores(all_tokens: list, att_scores: list) -> list:
    word_att_scores = []
    for i in range(len(all_tokens)):
        token, att_score = all_tokens[i], att_scores[i]
        word_att_scores.append([token, att_score])
    return word_att_scores

def clean_token(t):
    return t.replace("▁", "").replace("<0x0A>", "").strip()



def get_all_lines_score(word_att_scores: list, verified_flaw_lines: list, args):
    # === 清洗 verified flaw line 字串 ===
    verified_flaw_lines = [''.join([clean_token(tok) for tok in l]) for l in verified_flaw_lines]

    separator = ["<0x0A>"]
    all_lines_score = []
    flaw_line_indices = []

    score_sum = 0
    token_count = 0
    line_idx = 0
    line = ""

    for i in range(len(word_att_scores)):
        token, score = word_att_scores[i]

        if (token in separator or i == len(word_att_scores) - 1) and score_sum != 0:
            if token in separator:
                score_sum += score
                token_count += 1

            # === 根據選定模式計算該行 score ===
            if args.get_line_attn_mode == "sum":
                line_score = score_sum
            elif args.get_line_attn_mode == "mean":
                line_score = score_sum / token_count if token_count > 0 else 0
            elif args.get_line_attn_mode == "token_sqr_mean":
                line_score = score_sum / (token_count ** 2) if token_count > 0 else 0

            all_lines_score.append(line_score)

            # 比對這一行是否是缺陷行
            clean_line = line.replace("▁", "").replace("<0x0A>", "").strip()
            for v in verified_flaw_lines:
                if clean_line.strip() == v.strip():
                    flaw_line_indices.append(line_idx)
                    break

            # 重置
            line = ""
            score_sum = 0
            token_count = 0
            line_idx += 1

        elif token not in separator:
            line += token
            score_sum += score
            token_count += 1

    return all_lines_score, flaw_line_indices



def line_level_evaluation(all_lines_score: list, flaw_line_indices: list, top_k_loc: list, top_k_constant: list, true_positive_only: bool, index=None):
    if true_positive_only:    
        # line indices ranking based on attr values 
        ranking = sorted(range(len(all_lines_score)), key=lambda i: all_lines_score[i], reverse=True)
        # total flaw lines
        num_of_flaw_lines = len(flaw_line_indices)
        # clean lines + flaw lines
        total_lines = len(all_lines_score)
        ### TopK% Recall ###
        all_correctly_predicted_flaw_lines = []  
        ### IFA ###
        ifa = True
        all_clean_lines_inspected = []
        for top_k in top_k_loc:
            correctly_predicted_flaw_lines = 0
            for indice in flaw_line_indices:
                # if within top-k
                k = int(len(all_lines_score) * top_k)
                # if detecting any flaw lines
                if indice in ranking[: k]:
                    correctly_predicted_flaw_lines += 1
                if ifa:
                    # calculate Initial False Alarm
                    # IFA counts how many clean lines are inspected until the first vulnerable line is found when inspecting the lines ranked by the approaches.
                    flaw_line_idx_in_ranking = ranking.index(indice)
                    # e.g. flaw_line_idx_in_ranking = 3 will include 1 vulnerable line and 3 clean lines
                    all_clean_lines_inspected.append(flaw_line_idx_in_ranking)  
            # for IFA
            min_clean_lines_inspected = min(all_clean_lines_inspected)
            # for All Effort
            max_clean_lines_inspected = max(all_clean_lines_inspected)
            # only do IFA and All Effort once
            ifa = False
            # append result for one top-k value
            all_correctly_predicted_flaw_lines.append(correctly_predicted_flaw_lines)
        
        ### Top10 Accuracy ###
        all_correctly_localized_func = []
        top_10_correct_idx = []
        top_10_not_correct_idx = []
        correctly_located = False
        for k in top_k_constant:
            for indice in flaw_line_indices:
                # if detecting any flaw lines
                if indice in ranking[: k]:
                    """
                    # extract example for the paper
                    if index == 2797:
                        print("2797")
                        print("ground truth flaw line index: ", indice)
                        print("ranked line")
                        print(ranking)
                        print("original score")
                        print(all_lines_score)
                    """
                    # append result for one top-k value
                    all_correctly_localized_func.append(1)
                    correctly_located = True
                else:
                    all_correctly_localized_func.append(0)
            if correctly_located:
                top_10_correct_idx.append(index)
            else:
                top_10_not_correct_idx.append(index)
        return total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, \
               top_10_correct_idx, top_10_not_correct_idx
    else:
        # all_lines_score_with_label: [[line score, line level label], [line score, line level label], ...]
        all_lines_score_with_label = []
        for i in range(len(all_lines_score)):
            if i in flaw_line_indices:
                all_lines_score_with_label.append([all_lines_score[i], 1])
            else:
                all_lines_score_with_label.append([all_lines_score[i], 0])
        return all_lines_score_with_label



def top_k_effort(rank_df, sum_lines, sum_flaw_lines, top_k_loc, label_col_name="label"):
    target_flaw_line = int(sum_flaw_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += 1
        if rank_df[label_col_name][i] == 1:
            caught_flaw_line += 1
        if target_flaw_line == caught_flaw_line:
            break
    effort = round(inspected_line / sum_lines, 4)
    return effort, inspected_line

def top_k_recall(pos_rank_df, neg_rank_df, sum_lines, sum_flaw_lines, top_k_loc):
    target_inspected_line = int(sum_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    inspect_neg_lines = True
    for i in range(len(pos_rank_df)):
        inspected_line += 1
        if inspected_line > target_inspected_line:
            inspect_neg_lines = False
            break
        if pos_rank_df["label"][i] == 1 or pos_rank_df["label"][i] is True:
            caught_flaw_line += 1
    if inspect_neg_lines:
        for i in range(len(neg_rank_df)):
            inspected_line += 1
            if inspected_line > target_inspected_line:
                break
            if neg_rank_df["label"][i] == 1 or neg_rank_df["label"][i] is True:
                caught_flaw_line += 1
    return round(caught_flaw_line / sum_flaw_lines, 4)

def line_level_localization_tp(flaw_lines: str, tokenizer, model, mini_batch, original_func: str, args, top_k_loc: list, top_k_constant: list, reasoning_method: str, index: int, write_invalid_data: bool):
    flaw_line_seperator = "/~/"
    input_ids = mini_batch["input_ids"][0].unsqueeze(0).to(args.device)
    labels = mini_batch["label"][0].unsqueeze(0).to(args.device)
    attention_mask = mini_batch["attention_mask"][0].unsqueeze(0).to(args.device)
    ids = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    original_lines = ''.join(all_tokens).split("Ċ")
    input_ids = input_ids.to(args.device)

    # flaw line verification
    # get flaw tokens ground truth
    flaw_lines = get_all_flaw_lines(flaw_lines=flaw_lines, flaw_line_seperator=flaw_line_seperator)
    flaw_tokens_encoded = encode_all_lines(all_lines=flaw_lines, tokenizer=tokenizer)
    verified_flaw_lines = []
    do_explanation = False
    for i in range(len(flaw_tokens_encoded)):
        encoded_flaw = ''.join(flaw_tokens_encoded[i])
        encoded_all = ''.join(all_tokens)
        if encoded_flaw in encoded_all:
            verified_flaw_lines.append(flaw_tokens_encoded[i])
            do_explanation = True

    if do_explanation:
        if reasoning_method == "attention":
            # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
            input_ids = input_ids.to(args.device)
            loss, logits, attentions = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            # take from tuple then take out mini-batch attention values
            if isinstance(attentions[0], torch.Tensor) and attentions[0].dim() == 4:
                attentions = attentions[0][0]  # → list of (seq_len, seq_len)
            elif isinstance(attentions[0], torch.Tensor) and attentions[0].dim() == 3:
                attentions = attentions[0] # → list of (seq_len, seq_len)
            attention = None
            # go into the layer
            for i in range(len(attentions)):
                layer_attention = attentions[i]
                # summerize the values of each token dot other tokens
                layer_attention = sum(layer_attention)
                if attention is None:
                    attention = layer_attention
                else:
                    attention += layer_attention
            # clean att score for <s> and </s>
            attention = clean_special_token_values(all_tokens, attention)
            # attention should be 1D tensor with seq length representing each token's attention value
            word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)
            all_lines_score, flaw_line_indices = get_all_lines_score(word_att_scores, verified_flaw_lines, args)
            # return if no flaw lines exist
            if len(flaw_line_indices) == 0:
                return "NA"
            total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, top_10_correct_idx, top_10_not_correct_idx \
            = \
            line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices, top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=True, index=index)
        
        results = {"total_lines": total_lines,
        "num_of_flaw_lines": num_of_flaw_lines,
        "all_correctly_predicted_flaw_lines": all_correctly_predicted_flaw_lines,
        "all_correctly_localized_function": all_correctly_localized_func,
        "min_clean_lines_inspected": min_clean_lines_inspected,
        "max_clean_lines_inspected": max_clean_lines_inspected,
        "top_10_correct_idx": top_10_correct_idx,
        "top_10_not_correct_idx": top_10_not_correct_idx}
        return results
    else:
        if write_invalid_data:
            with open("../invalid_data/invalid_line_lev_data.txt", "a") as f:
                f.writelines("--- ALL TOKENS ---")
                f.writelines("\n")
                alltok = ''.join(all_tokens)
                alltok = alltok.split("Ċ")
                for tok in alltok:
                    f.writelines(tok)
                    f.writelines("\n")
                f.writelines("--- FLAW ---")
                f.writelines("\n")
                for i in range(len(flaw_tokens_encoded)):
                    f.writelines(''.join(flaw_tokens_encoded[i]))
                    f.writelines("\n")
                f.writelines("\n")
                f.writelines("\n")
    # if no flaw line exist in the encoded input
    return "NA"

# ========= Main =========
def main():
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a JSON file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="The input evaluation data file (a JSON file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="The input test data file (a JSON file).")

    parser.add_argument("--saved_model_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "test"],)
    parser.add_argument("--tokenizer", default="pretrained", choices=["pretrained", "non_pretrained"],
                        help="Optional pretrained tokenizer name or path, if not, the same as model_name_or_path")
    parser.add_argument("--model_name", default="codellama", choices=["codellama", "vulllama_b4", "vulllama_b8", "vulllama_b16", "vulllama_b24", "vulllama_b32"],
                        help="The model checkpoint for weights initialization")
    parser.add_argument("--reasoning_method", default="attention", choices=["attention", "shap", "lime", "lig"],
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")
    parser.add_argument("--get_line_attn_mode", default="sum", choices=["sum", "mean", "token_sqr_mean"],
                        help="How to get line attention score, should be one of 'sum', 'mean', 'token_sqr_mean'")
    parser.add_argument("--do_local_explanation", default=True, action='store_true',
                        help="Whether to do local explanation. ") 
    parser.add_argument("do_sorting_by_line_scores", default=True, action='store_true',
                        help="Whether to sort the line scores by attention or logits scores.")
    parser.add_argument("--effort_at_n_percent_recall", type=float, default=0.2,
                        help="Effort at n% recall, e.g., 0.2 means 20% recall")
    parser.add_argument("--recall_at_n_percent_loc", type=float, default=0.01,
                        help="Recall at n% LOC, e.g., 0.01 means 1% LOC")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="training epochs")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_output_path", type=str, default="./vul_llama/experiment/baseline/pred_result_from_main.csv")
    parser.add_argument("--log_file", type=str, default=None)

    args, _ = parser.parse_known_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log_file , mode='w'),
            logging.StreamHandler()
        ]
    )

    # Appened new log after last log
    def log_new_run_marker():
        logging.info("=" * 100)
        logging.info("🔄🔄🔄🔄🔄  NEW RUN STARTED  🔄🔄🔄🔄🔄")
        logging.info("=" * 100 + "\n")
    log_new_run_marker()


    # Checkup CUDA, GPU status
    logger.info(f"CUDA STATUS = {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu,)

    # === Load Dataset ===
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    train_data = load_json(args.train_data_file)
    eval_data = load_json(args.eval_data_file)
    test_data = load_json(args.test_data_file)

    # === Load Tokenizer ===
    logger.info("Loading tokenizer...")
    if args.tokenizer == "pretrained":
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLLaMA-7b-hf")
    elif args.tokenizer == "non_pretrained":    # BPE trained on AOSP
        tokenizer = AutoTokenizer.from_pretrained("./vul_llama/android_tokenizer_spm", use_fast=True)

    logger.info("Loading model...")
    if args.model_name == "codellama":
        model = VulLlamaBaseline()
    elif args.model_name.startswith("vulllama_b"):
        # 先偵測雙向層數
        match = re.search(r"vulllama_b(\d+)", args.model_name)
        bidir_layers = int(match.group(1)) if match else 0

        config = VulLlamaConfig.from_pretrained("codellama/CodeLLaMA-7b-hf")  # or custom path
        config.bidirectional_layer_count = bidir_layers  # ✅ 這邊設你想要的雙向層數
        base_model = VulLlamaModel(config)
        # base_model = VulLlamaModel.from_pretrained(args.resume_ckpt_path, config=config)
        if args.mode == "train":
            base_model.gradient_checkpointing_enable()  # ✅ 只在訓練模式啟用！
        model = FunctionLevelClassifier(base_model, hidden_size=config.hidden_size)

    logger.info("Training/evaluation parameters %s", args)
    if args.mode == "train":
        train_dataset = CodeFunctionDataset(json_data=train_data, tokenizer=tokenizer)
        eval_dataset = CodeFunctionDataset(json_data=eval_data, tokenizer=tokenizer)
        train(args, model, tokenizer, train_dataset, eval_dataset)
    if args.mode == "test":
        # 正確的 checkpoint 路徑
        if args.tokenizer == "pretrained":
            checkpoint_path = os.path.join(args.saved_model_dir, "checkpoint-best-f1", f"{args.model_name}.bin")
        elif args.tokenizer == "non_pretrained":
            checkpoint_path = os.path.join(args.saved_model_dir, "checkpoint-best-f1", f"{args.model_name}_{args.tokenizer}.pt")
        logger.info(f"🔁 Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device), strict=False)
        model.to(args.device)
        test_dataset = CodeFunctionDataset(json_data=test_data, tokenizer=tokenizer)
   
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)

if __name__ == "__main__":
    main()    
