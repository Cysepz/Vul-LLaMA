# ========= Vul Llama Model =========
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import logging
logger = logging.getLogger(__name__)


class VulLlamaConfig(LlamaConfig):
    def __init__(self, **kwargs):
        self.bidirectional_layer_count = kwargs.pop("bidirectional_layer_count", 0)
        super().__init__(**kwargs)

class VulLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.use_bidirectional = layer_idx < config.bidirectional_layer_count

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        position_embeddings=None,
    ):
        if self.use_bidirectional:
            attention_mask = None  # 移除 causal mask（變成雙向）
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

class VulLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: VulLlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            VulLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids=None, output_attentions=False):
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = self.embed_tokens(input_ids)
        dummy_cos_sin = (torch.zeros(1, device=hidden_states.device), torch.zeros(1, device=hidden_states.device))

        if attention_mask is not None:
            # 將 [B, L] 的 attention mask 擴展為 [B, 1, 1, L] 以符合 LLaMA 要求
            attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)  # (B, 1, 1, L)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min  # masked positions設為極小值

        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=dummy_cos_sin,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions.append(layer_outputs[1])  # 取出每層的 attention

        hidden_states = self.norm(hidden_states)
        if output_attentions:
            return hidden_states, all_attentions
        else:
            return (hidden_states,)


class FunctionLevelClassifier(nn.Module):
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, 2) # 輸出兩個 logits

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
         # Step 1: get hidden states from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden = outputs[0]  # shape: (B, L, H)

        # 若有 attention 傳回，抓出
        attentions = outputs[1] if len(outputs) > 1 else None

        # Step 2: extract [CLS] token or first token
        cls_token = last_hidden[:, 0, :]  # 取第一個 token 表徵

        # Step 3: classification logits
        logits = self.classifier(cls_token)  # shape: (B, 2)

        # Step 4: compute loss if labels are provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.long())
            return loss, logits, attentions  # ✅ 明確回傳 attention
        else:
            return None, logits, attentions  # ✅ 測試時會回這兩個