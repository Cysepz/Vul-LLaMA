import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import logging

# ========= Baseline Model =========
class VulLlamaBaseline(nn.Module):
    def __init__(self, model_name="codellama/CodeLlama-7b-hf"):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.llm.gradient_checkpointing_enable()    # Checkpointing for memory efficiency
        self.classifier = nn.Linear(self.llm.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, output_attentions=False):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True
        )

        # 舊的（CLS token）
        # cls_hidden = outputs.hidden_states[-1][:, 0, :]

        # ✅ 改成：最後一個非 padding token 的 hidden state
        last_token_indices = attention_mask.sum(dim=1) - 1  # 每筆輸入最後的有效 token 位置
        last_token_indices = torch.clamp(last_token_indices, min=0)
        batch_size = input_ids.size(0)
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]
        cls_hidden = last_hidden[torch.arange(batch_size), last_token_indices]
        
        cls_hidden = cls_hidden.to(self.classifier.weight.dtype)
        logits = self.classifier(cls_hidden)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, logits, outputs.attentions[-1] if output_attentions else None