import logging
from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


class LlamaSharedLayer(nn.Module):
    """
    Llama3.1 共享层，使用 LoRA 微调
    输出最后一层的 hidden states 作为下游任务的特征
    """

    def __init__(self, model_path: str, lora_config: dict, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device

        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载预训练模型
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # 配置 LoRA
        lora_config_obj = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("lora_r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none"
        )

        # 应用 LoRA
        self.model = get_peft_model(self.model, lora_config_obj)

        # 获取模型的隐藏层大小
        self.hidden_size = self.config.hidden_size

        logging.info(f"Llama model loaded with LoRA. Hidden size: {self.hidden_size}")

    def tokenize_inputs(self, texts: list, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        对输入文本进行tokenize
        """
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        返回最后一层的隐藏状态作为特征向量

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            hidden_states: [batch_size, hidden_size] - 最后一个token的隐藏状态
        """
        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # 获取最后一层的隐藏状态
            last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

            # 取最后一个有效token的隐藏状态（通常是EOS token）
            batch_size = input_ids.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1  # 最后一个有效token的位置

            # 收集每个序列最后一个token的隐藏状态
            pooled_hidden_states = last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

            return pooled_hidden_states  # [batch_size, hidden_size]

    def encode_text(self, texts: list, max_length: int = 512) -> torch.Tensor:
        """
        编码文本，返回特征向量

        Args:
            texts: 输入文本列表
            max_length: 最大长度

        Returns:
            features: [batch_size, hidden_size]
        """
        # Tokenize
        encoded = self.tokenize_inputs(texts, max_length)

        # Forward pass
        with torch.no_grad():
            features = self.forward(encoded["input_ids"], encoded["attention_mask"])

        return features

    def prepare_training_inputs(self, prompts: list, queries: list, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        准备训练输入，将prompt和query拼接

        Args:
            prompts: prompt列表
            queries: query列表
            max_length: 最大长度

        Returns:
            inputs: 包含input_ids和attention_mask的字典
        """
        # 拼接prompt和query
        combined_texts = [
            f"Question: {prompt}\nOption: {query}\nAnswer:"
            for prompt, query in zip(prompts, queries)
        ]

        return self.tokenize_inputs(combined_texts, max_length)

    def get_trainable_parameters(self):
        """
        获取可训练参数的数量
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return trainable_params, all_param

    def save_lora_weights(self, save_path: str):
        """
        保存LoRA权重
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logging.info(f"LoRA weights saved to {save_path}")

    def load_lora_weights(self, load_path: str):
        """
        加载LoRA权重
        """
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, load_path)
        logging.info(f"LoRA weights loaded from {load_path}")

