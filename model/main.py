import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM

from .args import ModelArgs
from .experts import ExpertLayer
from .router import ExpertRouter, AdaptiveRouter

from transformers import AutoConfig



class CulturalAlignmentModel(nn.Module):
    """
    文化对齐模型
    结合了Llama3.1共享层、路由算法和专家层的完整架构
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # 加载本地Llama3.1模型（内存优化版）
        logging.info(f"Loading Llama model from {args.llama_model_path}")
        # 自动加载与 Llama3.1 模型配套的分词器（tokenizer）
        # tokenizer 将文本（字符串）转换为模型可以处理的 token ID（整数序列），并生成 attention mask 等输入信息。
        self.tokenizer = AutoTokenizer.from_pretrained(args.llama_model_path)
        # 加载模型配置
        config = AutoConfig.from_pretrained(args.llama_model_path)

        # 修改 rope_scaling 配置
        # config.rope_scaling = {
        #     "type": "llama",  # 或者 "llama3"，根据需要设置
        #     "factor": 8.0
        # }

        # 优先尝试以8位量化（节省内存）方式加载Llama3.1模型，若失败则自动回退到float32精度加载
        try:
            # 首先尝试8位量化（如果可用）
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model_path,
                config=config,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
            )
            logging.info("Successfully loaded with 8-bit quantization")
        except Exception as e:
            logging.warning(f"8-bit quantization failed: {e}, trying alternative loading...")
            # 如果8位量化失败，使用CPU + float32
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model_path,
                config=config,
                torch_dtype=torch.float32,  # 使用float32可能更稳定
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logging.info("Successfully loaded with CPU + float32")

        # 设置padding token，确保tokenizer有一个有效的 padding token，以便对输入文本进行批量处理时能够正确填充（padding）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 配置Llama的LoRA
        llama_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules  # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        # 将Llama模型转换为LoRA模型
        self.llama_model = get_peft_model(self.llama_model, llama_lora_config)

        logging.info(f"Applied LoRA to Llama model with r={args.lora_r}, alpha={args.lora_alpha}")
        self.llama_model.print_trainable_parameters()

        # 获取Llama的隐藏层大小
        llama_hidden_size = self.llama_model.config.hidden_size  # 4096

        # 初始化路由器
        self.router = ExpertRouter( # 也可以调用 AdaptiveRouter
            input_dim=llama_hidden_size,
            num_experts=args.num_experts,
            hidden_dim=args.router_hidden_size,
            dropout=args.lora_dropout
        )

        # 初始化专家层
        self.expert_layer = ExpertLayer(
            input_dim=llama_hidden_size,
            expert_hidden_dim=args.expert_hidden_size,
            output_dim=args.expert_hidden_size,  # 专家输出维度
            num_experts=args.num_experts,
            dropout=args.lora_dropout,
            lora_rank=args.lora_r,
            lora_alpha=args.lora_alpha
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(args.expert_hidden_size, args.expert_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(args.lora_dropout),
            nn.Linear(args.expert_hidden_size // 2, args.num_classes)
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 初始化分类头权重
        self._init_classifier_weights()

        logging.info(f"Cultural Alignment Model initialized")
        logging.info(f"Llama hidden size: {llama_hidden_size}")
        logging.info(f"Number of experts: {args.num_experts}")
        logging.info(f"Expert hidden size: {args.expert_hidden_size}")

    def _init_classifier_weights(self):
        """初始化分类头权重"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播，训练阶段

        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size] (可选)

        Returns:
            outputs: 包含损失、logits和其他信息的字典
        """
        outputs = {}

        # 1. Llama LoRA模型提取特征（允许梯度计算）
        llama_outputs = self.llama_model(
            input_ids=input_ids, # 输入文本的token序列和掩码
            attention_mask=attention_mask,
            output_hidden_states=True # 要求模型返回所有层的hidden states
        )
        # 取最后一层hidden states的最后一个token作为句子表示（类似于CLS向量）
        llama_features = llama_outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
        outputs["llama_features"] = llama_features

        # 2. 路由算法计算专家权重，将Llama特征输入路由器，得到每个专家的权重分布
        expert_weights, router_logits = self.router(llama_features)  # 均为[batch_size, num_experts]
        outputs["expert_weights"] = expert_weights # 将 router_logits 经过 softmax 后得到的归一化权重分布
        outputs["router_logits"] = router_logits # 路由器的原始输出（未softmax前）

        # 3. 专家层处理
        expert_output, expert_info = self.expert_layer(llama_features, expert_weights)
        outputs["expert_output"] = expert_output # [batch_size, expert_hidden_size]，是所有专家输出的加权和
        outputs["expert_info"] = expert_info # 包含每个专家的单独输出和其他中间信息

        # 4. 分类
        logits = self.classifier(expert_output)  # [batch_size, num_classes]，MLP分类头将专家聚合特征映射到最终的类别空间
        outputs["logits"] = logits # 每一行为各类别的未归一化分数

        # 5. 计算损失
        if labels is not None:
            # 主任务损失（分类损失），交叉熵损失，衡量分类准确性
            classification_loss = self.criterion(logits, labels)
            outputs["classification_loss"] = classification_loss

            # 负载均衡损失，鼓励路由器均匀分配专家，防止只用少数专家
            load_balancing_loss = self.router.compute_load_balancing_loss(router_logits)
            outputs["load_balancing_loss"] = load_balancing_loss

            # 专家多样性损失，鼓励不同专家输出有差异（多样性），防止专家同质化
            diversity_loss = -self.expert_layer.compute_expert_diversity(llama_features)
            outputs["diversity_loss"] = diversity_loss

            # 总损失，三者加权和，权重分别为 1, 0.01, 0.001。
            total_loss = (classification_loss +
                         0.01 * load_balancing_loss +
                         0.001 * diversity_loss)
            outputs["loss"] = total_loss

        return outputs # 字典类型，包含所有中间特征、专家权重、分类logits、各项损失等

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测函数（推理阶段）

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码

        Returns:
            predictions: 预测结果字典
        """
        self.eval() # 切换模型到评估模式（evaluation mode），关闭 Dropout、BatchNorm 等训练时特有的行为
        with torch.no_grad(): # 不计算梯度（在预测时不需要反向传播，也不需要计算梯度）
            outputs = self.forward(input_ids, attention_mask) # 用 forward 方法进行一次完整的推理流程，得到包含 logits、专家权重等信息的输出字典

            # 获取预测
            logits = outputs["logits"] # [batch_size, num_classes]，分类头输出的未归一化分数
            probabilities = F.softmax(logits, dim=-1) # 对 logits 进行 softmax，得到每个样本属于各类别的概率分布
            predictions = torch.argmax(logits, dim=-1) # 在每个样本的 logits 上取最大值对应的索引，即为模型预测的类别

            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "logits": logits,
                "expert_weights": outputs["expert_weights"]
            } # 字典类型，包括类别标签、概率分布、原始分数及专家分配信息

    def encode_and_predict(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        端到端预测：从文本到预测结果

        Args:
            texts: 文本列表

        Returns:
            predictions: 预测结果
        """
        # 使用tokenizer编码文本，将输入的文本列表 texts 转换为模型可以接受的张量（tensor）格式
        inputs = self.tokenizer(
            texts,
            padding=True, # 自动补齐到统一长度
            truncation=True, # 超过最大长度则截断
            max_length=self.args.max_length, # 设置最大序列长度，防止过长输入
            return_tensors="pt" # 返回pytorch张量
        )

        # 移动到正确设备(4卡DDP)
        input_ids = inputs["input_ids"].to(self.args.device)
        attention_mask = inputs["attention_mask"].to(self.args.device)

        # 预测，调用模型的 predict 方法，进行前向推理，输出分类结果、概率分布、专家权重等信息
        return self.predict(input_ids, attention_mask)

    def get_expert_utilization_stats(self, dataloader) -> Dict[str, torch.Tensor]:
        """
        分析整个数据集在推理时，各专家的利用率情况，输出统计信息。

        Args:
            dataloader: 数据加载器

        Returns:
            stats: 专家利用率统计
        """
        self.eval() # 进入评估模式
        all_expert_weights = []

        with torch.no_grad(): # 不计算梯度
            for batch in dataloader: # 遍历数据集，对每个 batch，获取专家权重分布（每个样本对各专家的分配比例）
                input_ids = batch["input_ids"].to(self.args.device)
                attention_mask = batch["attention_mask"].to(self.args.device)

                outputs = self.forward(input_ids, attention_mask)
                all_expert_weights.append(outputs["expert_weights"])

        # 合并所有batch的权重，[总样本数, num_experts]，整个数据集所有样本的专家权重矩阵
        all_weights = torch.cat(all_expert_weights, dim=0)

        # 计算统计信息
        stats = {
            "mean_weights": all_weights.mean(dim=0), # 每个专家平均被分配的权重（反映专家整体利用率）
            "std_weights": all_weights.std(dim=0), # 每个专家分配权重标准差，反映分配是否均匀
            "max_weights": all_weights.max(dim=0)[0], # 每个专家被分配的最大权重
            "min_weights": all_weights.min(dim=0)[0], # 每个专家被分配的最小权重
            "utilization": self.router.get_expert_utilization(
                torch.cat([out["router_logits"] for out in
                          [self.forward(batch["input_ids"].to(self.args.device),
                                      batch["attention_mask"].to(self.args.device))
                           for batch in dataloader]], dim=0)
            )
        }

        return stats # 字典类型，用于分析专家分配是否均衡，是否有“死专家”或过度集中的现象

    def save_model(self, save_path: str):
        """
        保存模型

        Args:
            save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)

        # 保存模型状态（不包括Llama模型本身，因为它是冻结的）
        model_state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:  # 只保存可训练的参数（如LoRA、专家、路由器、分类头等）
                model_state[name] = param

        torch.save({
            'model_state_dict': model_state,
            'args': self.args,
        }, os.path.join(save_path, 'model.pt')) # 保存参数和配置到文件

        logging.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        加载模型，加载模型时，先加载Llama3.1主干，再加载你保存的LoRA/专家/路由器等可训练参数即可

        Args:
            load_path: 加载路径
        """
        checkpoint = torch.load(os.path.join(load_path, 'model.pt'), map_location=self.args.device)

        # 只加载可训练参数【Llama3.1 主干参数不需要重复保存和加载，在模型初始化时已经加载】
        model_state = checkpoint['model_state_dict']
        self.load_state_dict(model_state, strict=False)

        logging.info(f"Model loaded from {load_path}")

    def get_trainable_parameters(self):
        """
        统计模型各部分参数量及可训练参数比例，便于资源分析和调优
        """
        # Llama LoRA参数
        llama_total = sum(p.numel() for p in self.llama_model.parameters())
        llama_trainable = sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)

        # 路由器参数
        router_params = sum(p.numel() for p in self.router.parameters() if p.requires_grad)

        # 专家层参数
        expert_params = sum(p.numel() for p in self.expert_layer.parameters() if p.requires_grad)

        # 分类器参数
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)

        total_trainable = llama_trainable + router_params + expert_params + classifier_params
        total_params = llama_total + router_params + expert_params + classifier_params

        return {
            "llama_trainable": llama_trainable,
            "llama_total": llama_total,
            "router_params": router_params,
            "expert_params": expert_params,
            "classifier_params": classifier_params,
            "total_trainable": total_trainable,
            "total_params": total_params,
            "trainable_percentage": total_trainable / total_params * 100
        }

class CulturalAlignmentModelWithAdaptiveRouter(CulturalAlignmentModel):
    """
    使用自适应路由器的文化对齐模型
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)

        # 替换为自适应路由器
        llama_hidden_size = self.llama_model.config.hidden_size
        self.router = AdaptiveRouter(
            input_dim=llama_hidden_size,
            num_experts=args.num_experts,
            hidden_dim=args.router_hidden_size,
            dropout=args.lora_dropout
        )

        logging.info("Model initialized with adaptive router")
