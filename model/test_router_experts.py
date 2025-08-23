import torch
import torch.nn as nn
import torch.nn.functional as F
from router import ExpertRouter, AdaptiveRouter
from experts import ExpertLayer
from typing import Tuple


class RouterExpertIntegration(nn.Module):
    """
    路由器与专家层的集成模型
    """

    def __init__(self, router_input_dim: int, expert_input_dim: int, hidden_dim: int,
                 output_dim: int, num_experts: int = 6, router_type: str = "expert",
                 lora_rank: int = 8, dropout: float = 0.1):
        """
        初始化集成模型

        Args:
            router_input_dim: 路由器输入维度（通常与Llama隐藏层维度相同）
            expert_input_dim: 专家层输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_experts: 专家数量
            router_type: 路由器类型 ("expert" 或 "adaptive")
            lora_rank: LoRA秩
            dropout: dropout概率
        """
        super().__init__()

        self.router_input_dim = router_input_dim
        self.expert_input_dim = expert_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts

        self.router_type = router_type

        # 选择路由器类型
        if router_type == "expert":
            self.router = ExpertRouter(router_input_dim, num_experts, hidden_dim, dropout)
        elif router_type == "adaptive":
            self.router = AdaptiveRouter(router_input_dim, num_experts, hidden_dim, dropout)
        else:
            raise ValueError(f"未知的路由器类型: {router_type}")

        # 专家层
        self.expert_layer = ExpertLayer(expert_input_dim, hidden_dim, output_dim,
                                        num_experts, lora_rank, dropout)

    def forward(self, router_features: torch.Tensor, expert_features: torch.Tensor,
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            router_features: 路由器输入特征 [batch_size, router_input_dim]
            expert_features: 专家层输入特征 [batch_size, expert_input_dim]
            temperature: 温度参数（仅对ExpertRouter有效）

        Returns:
            output: 模型输出 [batch_size, output_dim]
            expert_weights: 专家权重 [batch_size, num_experts]
            router_logits: 路由logits [batch_size, num_experts]
        """
        # 路由器前向传播
        if self.router_type == "expert":
            expert_weights, router_logits = self.router(router_features, temperature)
        else:  # AdaptiveRouter
            expert_weights, router_logits, _ = self.router(router_features, temperature)

        # 专家层前向传播
        output, _ = self.expert_layer(expert_features, expert_weights)

        return output, expert_weights, router_logits

    def compute_loss(self, router_logits: torch.Tensor, target: torch.Tensor,
                     output: torch.Tensor, lb_weight: float = 0.01,
                     entropy_weight: float = 0.001) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失

        Args:
            router_logits: 路由logits
            target: 目标值
            output: 模型输出
            lb_weight: 负载均衡损失权重
            entropy_weight: 熵正则化损失权重

        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        # 主任务损失（这里使用MSE作为示例）
        task_loss = F.mse_loss(output, target)

        # 负载均衡损失
        lb_loss = self.router.compute_load_balancing_loss(router_logits)

        # 熵正则化损失
        expert_weights = F.softmax(router_logits, dim=-1)
        entropy_loss = self.router.entropy_regularization(expert_weights)

        # 总损失
        total_loss = task_loss + lb_weight * lb_loss + entropy_weight * entropy_loss

        # 损失字典
        loss_dict = {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "lb_loss": lb_loss,
            "entropy_loss": entropy_loss
        }

        return total_loss, loss_dict


def test_router_expert_integration():
    """测试路由器与专家层的集成"""
    print("=== 测试路由器与专家层集成 ===")

    # 设置参数
    router_input_dim = 512  # 路由器输入维度（来自Llama的隐藏层）
    expert_input_dim = 512  # 专家层输入维度
    hidden_dim = 256
    output_dim = 10
    num_experts = 6
    batch_size = 32
    lora_rank = 8
    dropout = 0.1

    # 创建集成模型（使用基础路由器）
    model = RouterExpertIntegration(
        router_input_dim=router_input_dim,
        expert_input_dim=expert_input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        router_type="expert",
        lora_rank=lora_rank,
        dropout=dropout
    )

    # 创建输入数据
    router_features = torch.randn(batch_size, router_input_dim)
    expert_features = torch.randn(batch_size, expert_input_dim)
    target = torch.randn(batch_size, output_dim)  # 假设的目标输出

    # 前向传播
    output, expert_weights, router_logits = model(router_features, expert_features)

    # 检查输出形状
    print(f"路由器输入形状: {router_features.shape}")
    print(f"专家层输入形状: {expert_features.shape}")
    print(f"专家权重形状: {expert_weights.shape}")
    print(f"路由logits形状: {router_logits.shape}")
    print(f"模型输出形状: {output.shape}")
    print(f"目标形状: {target.shape}")

    # 验证专家权重和为1
    weight_sums = expert_weights.sum(dim=1)
    print(f"专家权重和: {weight_sums}")
    print(f"权重和是否接近1: {torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)}")

    # 计算损失
    total_loss, loss_dict = model.compute_loss(router_logits, target, output)
    print(f"总损失: {total_loss.item():.6f}")
    print(f"任务损失: {loss_dict['task_loss'].item():.6f}")
    print(f"负载均衡损失: {loss_dict['lb_loss'].item():.6f}")
    print(f"熵正则化损失: {loss_dict['entropy_loss'].item():.6f}")

    # 测试专家利用率
    utilization = model.router.get_expert_utilization(router_logits)
    print(f"专家利用率: {utilization}")

    print("路由器与专家层集成测试通过!\n")


def test_adaptive_router_integration():
    """测试自适应路由器与专家层的集成"""
    print("=== 测试自适应路由器与专家层集成 ===")

    # 设置参数
    router_input_dim = 512
    expert_input_dim = 128
    hidden_dim = 256
    output_dim = 10
    num_experts = 6
    batch_size = 32
    lora_rank = 8
    dropout = 0.1

    # 创建集成模型（使用自适应路由器）
    model = RouterExpertIntegration(
        router_input_dim=router_input_dim,
        expert_input_dim=expert_input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        router_type="adaptive",
        lora_rank=lora_rank,
        dropout=dropout
    )

    # 创建输入数据
    router_features = torch.randn(batch_size, router_input_dim)
    expert_features = torch.randn(batch_size, expert_input_dim)
    target = torch.randn(batch_size, output_dim)

    # 前向传播
    output, expert_weights, router_logits = model(router_features, expert_features)

    # 检查输出形状
    print(f"路由器输入形状: {router_features.shape}")
    print(f"专家层输入形状: {expert_features.shape}")
    print(f"专家权重形状: {expert_weights.shape}")
    print(f"路由logits形状: {router_logits.shape}")
    print(f"模型输出形状: {output.shape}")

    # 验证专家权重和为1
    weight_sums = expert_weights.sum(dim=1)
    print(f"专家权重和: {weight_sums}")
    print(f"权重和是否接近1: {torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)}")

    # 计算损失
    total_loss, loss_dict = model.compute_loss(router_logits, target, output)
    print(f"总损失: {total_loss.item():.6f}")
    print(f"任务损失: {loss_dict['task_loss'].item():.6f}")
    print(f"负载均衡损失: {loss_dict['lb_loss'].item():.6f}")
    print(f"熵正则化损失: {loss_dict['entropy_loss'].item():.6f}")

    # 测试不同温度下的行为
    print("\n测试不同温度下的行为:")
    for temp in [0.5, 1.0, 2.0]:
        output_temp, expert_weights_temp, _ = model(router_features, expert_features, temperature=temp)

        # 计算权重分布的熵
        entropy = -torch.sum(expert_weights_temp * torch.log(expert_weights_temp + 1e-8), dim=-1).mean()
        print(f"温度 {temp}: 平均熵 = {entropy.item():.4f}")

    print("自适应路由器与专家层集成测试通过!\n")


def test_training_step():
    """测试训练步骤"""
    print("=== 测试训练步骤 ===")

    # 设置参数
    router_input_dim = 512
    expert_input_dim = 128
    hidden_dim = 256
    output_dim = 10
    num_experts = 6
    batch_size = 32
    lora_rank = 8
    dropout = 0.1

    # 创建模型
    model = RouterExpertIntegration(
        router_input_dim=router_input_dim,
        expert_input_dim=expert_input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        router_type="expert",
        lora_rank=lora_rank,
        dropout=dropout
    )

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模拟训练步骤
    for step in range(3):  # 模拟3个训练步骤
        # 创建输入数据
        router_features = torch.randn(batch_size, router_input_dim)
        expert_features = torch.randn(batch_size, expert_input_dim)
        target = torch.randn(batch_size, output_dim)

        # 前向传播
        output, expert_weights, router_logits = model(router_features, expert_features)

        # 计算损失
        total_loss, loss_dict = model.compute_loss(router_logits, target, output)

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"步骤 {step + 1}: 总损失 = {total_loss.item():.6f}")

    print("训练步骤测试完成!\n")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 运行测试
    test_router_expert_integration()
    test_adaptive_router_integration()
    test_training_step()

    print("所有集成测试完成!")