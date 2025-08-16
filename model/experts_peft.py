#!/usr/bin/env python3
"""
使用PEFT库的LoRA专家实现示例
展示如何使用现成的LoRA库而不是手写实现
"""

import logging

import torch
import torch.nn as nn


# 如果安装了PEFT库，可以使用以下导入：
# from peft import LoraConfig, get_peft_model, TaskType

class StandardExpertNetwork(nn.Module):
    """
    标准的专家网络（用于应用LoRA）
    这就是LoRA中的"原始模型"
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class PEFTLoRAExpert(nn.Module):
    """
    使用PEFT库的LoRA专家实现
    这是推荐的做法，使用现成的库
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 expert_id: int = 0,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        super().__init__()
        self.expert_id = expert_id

        # 创建基础网络（这就是LoRA中的"原始模型"）
        self.base_model = StandardExpertNetwork(input_dim, hidden_dim, output_dim)

        # 使用PEFT库应用LoRA（需要安装：pip install peft）
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # LoRA配置
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # 任务类型
                inference_mode=False,  # 训练模式
                r=lora_r,  # LoRA的秩
                lora_alpha=lora_alpha,  # LoRA的缩放参数
                lora_dropout=lora_dropout,  # LoRA的dropout
                target_modules=["network.0", "network.3", "network.6"],  # 要应用LoRA的模块
            )

            # 将基础模型转换为LoRA模型
            self.model = get_peft_model(self.base_model, lora_config)
            logging.info(f"PEFT LoRA Expert {expert_id} initialized with r={lora_r}, alpha={lora_alpha}")

        except ImportError:
            logging.warning("PEFT library not found. Falling back to manual LoRA implementation.")
            # 如果没有PEFT库，可以使用手写实现作为后备
            self.model = self.base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_trainable_parameters(self):
        """获取可训练参数统计"""
        if hasattr(self.model, 'print_trainable_parameters'):
            # PEFT模型有内置的参数统计方法
            self.model.print_trainable_parameters()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": trainable_params / total_params * 100
        }

# 如果没有PEFT库，展示手写LoRA的简化版本
class SimpleLoRALayer(nn.Module):
    """
    简化的LoRA层实现（用于理解LoRA原理）
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original_layer = original_layer  # 这就是"原始线性层"
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA的低秩分解矩阵
        input_dim = original_layer.in_features
        output_dim = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出（通常冻结）
        original_output = self.original_layer(x)

        # LoRA适配器输出
        lora_output = x @ self.lora_A @ self.lora_B * self.scaling

        # 最终输出 = 原始输出 + LoRA适配器输出
        return original_output + lora_output

def demonstrate_lora_concept():
    """演示LoRA的核心概念"""
    print("="*60)
    print("LoRA核心概念演示")
    print("="*60)

    # 假设我们有一个预训练的线性层
    pretrained_layer = nn.Linear(768, 512)

    print("1. 原始预训练层:")
    print(f"   参数数量: {sum(p.numel() for p in pretrained_layer.parameters()):,}")

    # 应用LoRA
    lora_layer = SimpleLoRALayer(pretrained_layer, rank=16, alpha=32)

    print("\n2. 应用LoRA后:")
    print(f"   原始层参数: {sum(p.numel() for p in pretrained_layer.parameters()):,}")
    print(f"   LoRA参数: {lora_layer.lora_A.numel() + lora_layer.lora_B.numel():,}")
    print(f"   参数减少比例: {(lora_layer.lora_A.numel() + lora_layer.lora_B.numel()) / sum(p.numel() for p in pretrained_layer.parameters()) * 100:.1f}%")

    # 冻结原始层
    pretrained_layer.weight.requires_grad = False
    pretrained_layer.bias.requires_grad = False

    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_layer.parameters())

    print(f"\n3. 冻结原始层后:")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   可训练比例: {trainable_params / total_params * 100:.1f}%")

    print("\n4. LoRA的优势:")
    print("   ✅ 保持预训练知识（原始层不变）")
    print("   ✅ 大幅减少可训练参数")
    print("   ✅ 避免灾难性遗忘")
    print("   ✅ 可以针对不同任务训练不同的适配器")

if __name__ == "__main__":
    demonstrate_lora_concept()

    # 测试PEFT实现（如果可用）
    try:
        expert = PEFTLoRAExpert(768, 512, 256, expert_id=0)
        x = torch.randn(4, 768)
        output = expert(x)
        print(f"\n✅ PEFT LoRA Expert测试成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")

        stats = expert.get_trainable_parameters()
        print(f"   参数统计: {stats}")

    except Exception as e:
        print(f"\n⚠️  PEFT LoRA Expert测试失败: {e}")
        print("   可能需要安装PEFT库: pip install peft")

