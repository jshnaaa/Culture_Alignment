import torch
from experts import ExpertLayer


def test_expert_layer():
    # 设置一些测试参数
    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    num_experts = 6  # 使用6个专家
    lora_rank = 8
    dropout = 0.1
    batch_size = 32  # 测试时使用32个样本

    # 创建输入数据
    features = torch.randn(batch_size, input_dim)
    expert_weights = torch.randn(batch_size, num_experts)  # 假设的专家权重，大小为 [batch_size, num_experts]

    # 初始化ExpertLayer
    expert_layer = ExpertLayer(experts_input_dim=input_dim,
                               experts_hidden_dim=hidden_dim,
                               experts_output_dim=output_dim,
                               num_experts=num_experts,
                               lora_rank=lora_rank,
                               dropout=dropout)

    # 前向传播
    weighted_output, expert_outputs = expert_layer(features, expert_weights)

    # 打印每个专家的输出形状
    for expert_id, expert_output in enumerate(expert_outputs):
        print(f"Expert {expert_id} 输出形状: {expert_output.shape}")

    # 检查输出形状
    print("加权输出形状:", weighted_output.shape)  # 期望输出形状是 [batch_size, output_dim]

    # 确认输出符合预期
    assert weighted_output.shape == (batch_size, output_dim), "输出形状不符合预期！"


if __name__ == "__main__":
    test_expert_layer()
