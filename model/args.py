from dataclasses import dataclass, field  # 导入 field
from typing import Dict, Any

import torch


@dataclass
class ModelArgs:
    # Dataset configuration
    dataset_type: str = "globalopinions"  # "culturalbench", "globalopinions", "culturebank"

    # Model paths
    llama_model_path: str = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"

    # Model architecture
    num_experts: int = 6
    experts_hidden_size: int = 512
    experts_output_dim: int = 256
    router_hidden_size: int = 512
    num_classes: int = 2  # TRUE/FALSE classification for culturalbench

    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["all"]) # 目标模块的选择将由 LoRA 的实现自动决定，=all：LoRA 将尝试在模型中的所有可支持的线性层上应用 LoRA

    # Training parameters
    learning_rate: float = 1e-6
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data parameters
    max_length: int = 512
    # train_data_path: str = "/root/autodl-fs/CulturalBench-Hard_train.json"
    # test_data_path: str = "/root/autodl-fs/CulturalBench-Hard_test.json"
    train_data_path: str = "/root/autodl-fs/global_opinions_train.json"
    test_data_path: str = "/root/autodl-fs/global_opinions_test.json"

    # Device and training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

    # 损失权重
    load_balance_weight: float = 0.01
    diversity_weight: float = 0.001

    # Output paths
    output_dir: str = "/root/autodl-tmp/CultureMoE/Culture_Alignment/outputs/" + dataset_type
    model_save_path: str = output_dir + "/CA_llama"
    log_file: str = output_dir + "/CA_llama_training.log"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # 根据数据集类型自动配置参数
        self._configure_dataset_specific_params()

    def _configure_dataset_specific_params(self):
        """根据数据集类型配置特定参数"""
        if self.dataset_type == "culturalbench":
            self.num_classes = 2  # TRUE/FALSE binary classification
            self.loss_type = "classification"
            self.output_type = "classification"

        elif self.dataset_type == "globalopinions":
            self.num_classes = 5  # ['Very favorable', 'Somewhat favorable', 'Somewhat unfavorable', 'Very unfavorable', 'DK/Refused']
            self.loss_type = "js_divergence"
            self.output_type = "probability_distribution"

        elif self.dataset_type == "culturebank":
            # 预留给未来的数据集
            self.num_classes = None  # 待定义
            self.loss_type = "to_be_defined"
            self.output_type = "to_be_defined"

        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def get_dataset_config(self) -> Dict[str, Any]:
        """获取当前数据集的配置信息"""
        return {
            "dataset_type": self.dataset_type,
            "num_classes": self.num_classes,
            "loss_type": self.loss_type,
            "output_type": self.output_type
        }

