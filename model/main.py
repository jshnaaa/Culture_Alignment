import logging
import os
from typing import Dict, List, Optional
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .args import ModelArgs
from .experts import ExpertLayer
from .router import ExpertRouter, AdaptiveRouter

from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig

import torch.distributed as dist


class CulturalAlignmentModel(nn.Module):
    """
    文化对齐模型: Llama3.1共享层、路由算法和专家层
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # 获取模型路径
        if isinstance(args.llama_model_path, str):
            self.model_path = args.llama_model_path
        else:
            self.model_path = getattr(args.llama_model_path, 'llama_model_path', None)
            if self.model_path is None:
                raise ValueError("Provided object does not have 'llama_model_path' attribute")

        # 检查是否在分布式环境中
        self.is_distributed = hasattr(args, 'distributed') and args.distributed
        
        # 加载本地Llama3.1模型
        logging.info(f"Loading Llama model from {args.llama_model_path}")

        # 自动加载与 Llama3.1 模型配套的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(args.llama_model_path)

        # 加载模型配置
        config = AutoConfig.from_pretrained(args.llama_model_path)

        # 根据是否分布式训练选择不同的加载方式
        if self.is_distributed:
            # 分布式训练模式 - 不使用 device_map 和量化
            logging.info("Using distributed training mode - disabling device_map and quantization")
            
            # 设置加载参数
            load_kwargs = {
                "config": config,
                "torch_dtype": torch.float16,  # 使用float16节省显存
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "device_map": None,  # 禁用自动设备映射
            }
            
            try:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )
                # 将模型移动到当前设备
                self.llama_model = self.llama_model.to(args.device)
                logging.info("Successfully loaded model for distributed training")
            except Exception as e:
                logging.error(f"Failed to load model for distributed training: {e}")
                raise
        else:
            # 单机训练模式 - 可以使用量化和设备映射
            logging.info("Using single-GPU training mode - enabling quantization if available")
            
            try:
                # 首先尝试8位量化（如果可用）
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    config=config,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config,
                    offload_folder="./offload"
                )
                logging.info("Successfully loaded with 8-bit quantization")
            except Exception as e:
                logging.warning(f"8-bit quantization failed: {e}, trying alternative loading...")
                # 如果8位量化失败，使用float16精度加载
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    config=config,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                logging.info("Successfully loaded with float16 precision")

        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 配置Llama的LoRA
        llama_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules if args.target_modules != ["all"] else None
        )

        # 将Llama模型转换为LoRA模型
        self.llama_model = get_peft_model(self.llama_model, llama_lora_config)
        
        # 在分布式模式下，确保模型在当前设备上
        if self.is_distributed:
            self.llama_model = self.llama_model.to(args.device)

        logging.info(f"Applied LoRA to Llama model with r={args.lora_r}, alpha={args.lora_alpha}")
        self.llama_model.print_trainable_parameters()

        # 获取Llama的隐藏层大小
        llama_hidden_size = self.llama_model.config.hidden_size

        # 初始化路由器
        self.router = ExpertRouter(
            router_input_dim=llama_hidden_size,
            num_experts=args.num_experts,
            router_hidden_dim=args.router_hidden_size,
            dropout=args.lora_dropout
        )

        self.expert_layer = ExpertLayer(
            experts_input_dim=llama_hidden_size,
            experts_hidden_dim=args.experts_hidden_size,
            experts_output_dim=args.experts_output_dim,
            num_experts=args.num_experts,
            lora_rank=args.lora_r,
            dropout=args.lora_dropout
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(args.experts_output_dim, args.experts_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.lora_dropout),
            nn.Linear(args.experts_output_dim // 2, args.num_classes)
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 获取数据集配置
        self.dataset_config = args.get_dataset_config()

        # 初始化分类头权重
        self._init_classifier_weights()

        logging.info(f"Cultural Alignment Model initialized")
        logging.info(f"Dataset type: {self.dataset_config['dataset_type']}")
        logging.info(f"Loss type: {self.dataset_config['loss_type']}")
        logging.info(f"Output type: {self.dataset_config['output_type']}")
        logging.info(f"Llama hidden size: {llama_hidden_size}")
        logging.info(f"Number of experts: {args.num_experts}")
        logging.info(f"Expert hidden size: {args.experts_hidden_size}")

        # 确保所有组件使用相同的dtype
        llama_dtype = next(self.llama_model.parameters()).dtype
        self.router = self.router.to(dtype=llama_dtype)
        self.expert_layer = self.expert_layer.to(dtype=llama_dtype)
        self.classifier = self.classifier.to(dtype=llama_dtype)
        
        logging.info(f"所有组件使用dtype: {llama_dtype}")

    def _init_classifier_weights(self):
        """初始化分类头权重"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def js_divergence_loss(self, pred_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """
        计算JS散度损失

        Args:
            pred_probs: 预测的概率分布 [batch_size, num_classes]
            target_probs: 目标概率分布 [batch_size, num_classes]

        Returns:
            js_loss: JS散度损失
        """
        # 确保概率分布归一化
        pred_probs = F.softmax(pred_probs, dim=-1)
        target_probs = F.softmax(target_probs, dim=-1)

        # 计算中间分布M = (P + Q) / 2
        m = (pred_probs + target_probs) / 2

        # 计算KL散度 KL(P||M) 和 KL(Q||M)
        kl_pm = F.kl_div(torch.log(pred_probs + 1e-8), m, reduction='none').sum(dim=-1)
        kl_qm = F.kl_div(torch.log(target_probs + 1e-8), m, reduction='none').sum(dim=-1)

        # JS散度 = 0.5 * (KL(P||M) + KL(Q||M))
        js_divergence = 0.5 * (kl_pm + kl_qm)

        # 直接返回JS散度作为损失（越小表示分布越相似）
        return js_divergence.mean()

    def parse_global_opinions_target(self, target_data: List[str]) -> torch.Tensor:
        """
        安全解析global opinions数据集的目标数据

        Args:
            target_data: 字符串形式的目标数据列表

        Returns:
            target_probs: 解析后的概率分布 [batch_size, num_classes]
        """
        batch_size = len(target_data)
        target_probs = torch.zeros(batch_size, self.args.num_classes, device=self.args.device)

        for i, target_str in enumerate(target_data):
            try:
                # 使用json安全解析
                if isinstance(target_str, str):
                    # 提取字典部分（移除defaultdict包装）
                    dict_start = target_str.find('{')
                    dict_end = target_str.rfind('}') + 1
                    if dict_start >= 0 and dict_end > dict_start:
                        dict_str = target_str[dict_start:dict_end]
                        target_dict = json.loads(dict_str.replace("'", '"'))
                    else:
                        raise ValueError("No dictionary found in target string")
                else:
                    target_dict = target_str

                # 合并所有国家的概率（取平均）
                country_probs = []
                for country, probs in target_dict.items():
                    if isinstance(probs, list) and len(probs) == self.args.num_classes:
                        country_probs.append(probs)

                if country_probs:
                    # 计算所有国家的平均概率
                    avg_probs = np.mean(country_probs, axis=0)
                    target_probs[i] = torch.tensor(avg_probs, dtype=torch.float32, device=self.args.device)
                else:
                    # 如果没有有效数据，使用均匀分布
                    target_probs[i] = torch.ones(self.args.num_classes, device=self.args.device) / self.args.num_classes

            except Exception as e:
                logging.warning(f"Failed to parse target data at index {i}: {e}")
                # 如果解析失败，使用均匀分布
                target_probs[i] = torch.ones(self.args.num_classes, device=self.args.device) / self.args.num_classes

        return target_probs


    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                target_probs: Optional[torch.Tensor] = None,
                raw_responses: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
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
        outputs["expert_output"] = expert_output # [batch_size, experts_hidden_size]，是所有专家输出的加权和
        outputs["expert_info"] = expert_info # 包含每个专家的单独输出和其他中间信息

        # 4. 分类
        logits = self.classifier(expert_output)  # [batch_size, num_classes]，MLP分类头将专家聚合特征映射到最终的类别空间
        outputs["logits"] = logits # 每一行为各类别的未归一化分数

        # 5. 计算损失
        if labels is not None or target_probs is not None or raw_responses is not None:
            # 根据数据集类型计算主任务损失
            if self.dataset_config["loss_type"] == "classification":
                # CulturalBench数据集：分类损失
                if labels is not None:
                    classification_loss = self.criterion(logits, labels)
                else:
                    raise ValueError("Labels required for classification loss")

            elif self.dataset_config["loss_type"] == "js_divergence":
                # Global Opinions数据集：JS散度损失
                if target_probs is not None:
                    classification_loss = self.js_divergence_loss(logits, target_probs)
                elif raw_responses is not None:
                    # 解析原始响应数据
                    parsed_target_probs = self.parse_global_opinions_target(raw_responses)
                    classification_loss = self.js_divergence_loss(logits, parsed_target_probs)
                else:
                    raise ValueError("Target probabilities or raw responses required for JS divergence loss")

            else:
                raise ValueError(f"Unknown loss type: {self.dataset_config['loss_type']}")

            outputs["classification_loss"] = classification_loss

            # 负载均衡损失，鼓励路由器均匀分配专家，防止只用少数专家
            load_balancing_loss = self.router.compute_load_balancing_loss(router_logits)
            outputs["load_balancing_loss"] = load_balancing_loss

            # 专家多样性损失，鼓励不同专家输出有差异（多样性），防止专家同质化
            diversity_loss = -self.expert_layer.compute_expert_diversity(llama_features)
            outputs["diversity_loss"] = diversity_loss

            # 总损失，三者加权和，权重分别为 1, 0.01, 0.001。
            total_loss = (classification_loss +
                          self.args.load_balance_weight * load_balancing_loss +
                          self.args.diversity_weight * diversity_loss)
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
            'llama_model_state_dict': self.llama_model.state_dict(),  # 保存Llama模型
            'args': self.args,
        }, os.path.join(save_path, 'model.pt'))
        # 保存参数和配置到文件

        logging.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        加载模型，加载模型时，先加载Llama3.1主干，再加载你保存的LoRA/专家/路由器等可训练参数即可

        Args:
            load_path: 加载路径
        """
        checkpoint = torch.load(os.path.join(load_path, 'model.pt'), map_location=self.args.device)
        self.llama_model.load_state_dict(checkpoint['llama_model_state_dict'])  # 恢复Llama模型
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

