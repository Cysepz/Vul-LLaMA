# Vul LLaMA: Enhancing Line-Level Vulnerability Localization Performance in Android Code by Fine-Tuning the Self-Attention Mechanism of Code LLaMA
## Overview

**VulLLaMA** is a replication-ready research prototype that fine-tunes the self-attention mechanism in Code LLaMA for line-level vulnerability prediction in Android source code. 


This repository contains:

* Data preprocessing scripts
* Model training, evaluation, testing routines
* Evaluation metrics & result files

> This repository is meant to accompany an academic paper submission. Please refer to the paper for more context.

---

## Repository Structure

```
├── data/                   # Dataset folder (not tracked by Git; ignored in .gitignore)
├── vul_llama/              # Main source code (model architecture, trainer, etc.)
    ├── train_bpe_tokenizer.py          # Clone AOSP, make corpus, merge AOSP corpus and code llama vocab to train bpe' tokenizer
    ├── cl_model.py                     # Define Code LLaMA model arch
    ├── vl_model.py                     # Define Vul LLaMA model
    ├── main.py                         # main execute file

├── saved_model/            # Fine-tuned model checkpoints (ignored)
├── statistic
    ├── generate_rank.py                # Rank result of Vul LLaMA / LineVul model
    ├── generate_rank.py                # Rank result of Vul LLaMA / LineVul model

├── environment.yml         # Conda environment setup
├── .gitignore              # Files/folders excluded from Git tracking
├── README.md               # This file
```

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/VulLLaMA.git
cd VulLLaMA
```

### Step 2: Set Up the Environment

We recommend using conda:

```bash
conda env create -f environment.yml
conda activate venv
```

### Step 3: Download and Unzip Dataset
```bash
cd VulLLaMA/data

# Download Dataset
wget https://github.com/TUTUTU0817/CodeLLaMA_GCN_model/blob/main/Dataset/raw_dataset.tar.gz

# Unzip Dataset
unzip merged_output_function.zip raw_dataset.json

# Process, balance and split dataset
python ./pre_process.py --mode process
python ./pre_process.py --mode balance
python ./pre_process.py --mode split
```
All of the dataset has the same number of columns (i.e., 39 cols), we focus on the following 3 columns to conduct our experiments:

Place the following files under `data/`:

* `processed_dataset.json`
* `raw_dataset.json`
* `balanced_dataset.json`

---

## Usage

### Train BPE on AOSP
```bash
cd  VulLLaMA/vul_llama
python ./train_bpe_tokenizer.py
```

### Train Code LLaMA
```bash
python ./vul_llama/main.py \
         --log_file ./vul_llama/train_logs/baseline_train.log \
         --train_data_file ./data/train.json --test_data_file ./data/test.json --eval_data_file ./data/eval.json \
         --mode train --tokenizer non_pretrained --model_name codellama \
         --saved_model_dir ./saved_model \
         --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 1e-4 --weight_decay 0.0 --adam_epsilon 1e-3 --max_grad_norm 1.0 --epochs 18 \
         --device cuda
```

### Test Code LLaMA
```bash
python ./vul_llama/main.py \
         --log_file ./vul_llama/train_logs/baseline_test.log \
         --train_data_file ./data/train.json --test_data_file ./data/test.json --eval_data_file ./data/eval.json \
         --mode test --tokenizer pretrained --model_name codellama \
         --saved_model_dir ./saved_model \
         --test_output_path ./vul_llama/results/baseline_pred_result.csv \
         --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 1e-4 --weight_decay 0.0 --adam_epsilon 1e-3 --max_grad_norm 1.0 --epochs 18 \
         --device cuda
```

### Train VulLLaMA
```bash
python ./vul_llama/main.py \
         --log_file ./vul_llama/train_logs/vulllama_train_b4.log \
         --train_data_file ./data/train.json --test_data_file ./data/test.json --eval_data_file ./data/eval.json \
         --mode train --tokenizer non_pretrained --model_name vulllama_b4 \
         --saved_model_dir ./saved_model \
         --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 1e-4 --weight_decay 0.0 --adam_epsilon 1e-3 --max_grad_norm 1.0 --epochs 18 \
         --device cuda
```

### Test VulLLaMA
```bash
python ./vul_llama/main.py \
         --log_file ./vul_llama/train_logs/vulllama_test_b4.log \
         --train_data_file ./data/train.json --test_data_file ./data/test.json --eval_data_file ./data/eval.json \
         --mode test --tokenizer non_pretrained --model_name vulllama_b4 \
         --saved_model_dir ./saved_model \
	     --get_line_attn_mode mean \
         --test_output_path ./vul_llama/results/vulllama_b4_pred_result_mean.csv \
         --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 1e-4 --weight_decay 0.0 --adam_epsilon 1e-3 --max_grad_norm 1.0 --epochs 18 \
         --device cuda
```

> You may customize training parameters in `run_vulllama.py` as needed.

---

## Acknowledgements

* Code LLaMA developers
* LineVul authors and dataset
* HuggingFace & PyTorch teams

---

## Citation

If you use VulLLaMA in your work, please cite our paper (to be added upon acceptance).
