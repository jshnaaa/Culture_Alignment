from dataclasses import dataclass

import torch


@dataclass
class ModelArgs:
    # Model paths
    llama_model_path: str = "Meta-Llama-3.1-8B-Instruct"

    # Model architecture
    num_experts: int = 6
    expert_hidden_size: int = 512
    router_hidden_size: int = 512
    num_classes: int = 2  # TRUE/FALSE classification

    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None # 目标模块的选择将由 LoRA 的实现自动决定，=all：LoRA 将尝试在模型中的所有可支持的线性层上应用 LoRA

    # Training parameters
    learning_rate: float = 5e-5
    num_epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data parameters
    max_length: int = 512
    train_data_path: str = "data/CulturalBench-Hard_train.json"
    test_data_path: str = "data/CulturalBench-Hard_test.json"

    # Device and training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

    # Output paths
    output_dir: str = "outputs"
    model_save_path: str = "outputs/best_model"
    log_file: str = "outputs/training.log"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

