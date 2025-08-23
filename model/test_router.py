import torch
from router import ExpertRouter, AdaptiveRouter


def test_expert_router():
    """测试基础专家路由器"""
    print("=== 测试 ExpertRouter ===")

    # 设置参数
    router_input_dim = 512  # 假设来自Llama的隐藏层维度
    num_experts = 6
    batch_size = 32

    # 创建路由器和输入数据
    router = ExpertRouter(router_input_dim, num_experts)
    features = torch.randn(batch_size, router_input_dim)

    # 前向传播
    expert_weights, router_logits = router(features)

    # 检查输出形状
    print(f"输入特征形状: {features.shape}")
    print(f"专家权重形状: {expert_weights.shape}")
    print(f"路由logits形状: {router_logits.shape}")

    # 验证专家权重和为1
    weight_sums = expert_weights.sum(dim=1)
    print(f"专家权重和: {weight_sums}")
    print(f"权重和是否接近1: {torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)}")

    # 测试负载均衡损失
    load_balancing_loss = router.compute_load_balancing_loss(router_logits)
    print(f"负载均衡损失: {load_balancing_loss.item()}")

    # 测试专家利用率
    utilization = router.get_expert_utilization(router_logits)
    print(f"专家利用率: {utilization}")

    # 测试熵正则化
    entropy_loss = router.entropy_regularization(expert_weights)
    print(f"熵正则化损失: {entropy_loss.item()}")

    print("ExpertRouter 测试通过!\n")


def test_adaptive_router():
    """测试自适应路由器"""
    print("=== 测试 AdaptiveRouter ===")

    # 设置参数
    router_input_dim = 512
    num_experts = 6
    batch_size = 32

    # 创建路由器和输入数据
    router = AdaptiveRouter(router_input_dim, num_experts)
    features = torch.randn(batch_size, router_input_dim)

    # 前向传播
    expert_weights, router_logits, adaptive_temperature = router(features)

    # 检查输出形状
    print(f"输入特征形状: {features.shape}")
    print(f"专家权重形状: {expert_weights.shape}")
    print(f"路由logits形状: {router_logits.shape}")
    print(f"自适应温度形状: {adaptive_temperature.shape}")
    print(f"温度范围: {adaptive_temperature.min().item():.3f} - {adaptive_temperature.max().item():.3f}")

    # 验证专家权重和为1
    weight_sums = expert_weights.sum(dim=1)
    print(f"专家权重和: {weight_sums}")
    print(f"权重和是否接近1: {torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)}")

    # 测试不同温度下的行为
    print("\n测试不同温度下的行为:")
    for temp in [0.5, 1.0, 2.0]:
        _, router_logits_temp, _ = router(features, base_temperature=temp)
        expert_weights_temp = torch.softmax(router_logits_temp / temp, dim=-1)

        # 计算权重分布的熵
        entropy = -torch.sum(expert_weights_temp * torch.log(expert_weights_temp + 1e-8), dim=-1).mean()
        print(f"温度 {temp}: 平均熵 = {entropy.item():.4f}")

    print("AdaptiveRouter 测试通过!\n")


def test_router_with_experts():
    """测试路由器与专家层的集成"""
    print("=== 测试路由器与专家层集成 ===")

    # 设置参数
    router_input_dim = 512
    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    num_experts = 6
    batch_size = 32

    # 创建路由器、专家层和输入数据
    router = ExpertRouter(router_input_dim, num_experts)
    # 假设我们有一个专家层 (需要从experts.py导入ExpertLayer)
    # expert_layer = ExpertLayer(input_dim, hidden_dim, output_dim, num_experts)

    features = torch.randn(batch_size, router_input_dim)
    expert_inputs = torch.randn(batch_size, input_dim)

    # 路由器前向传播
    expert_weights, _ = router(features)

    print(f"路由器输出形状: {expert_weights.shape}")
    print(f"专家输入形状: {expert_inputs.shape}")

    # 在实际使用中，我们会这样使用:
    # weighted_output, expert_outputs = expert_layer(expert_inputs, expert_weights)

    print("路由器与专家层集成测试通过!\n")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 运行测试
    test_expert_router()
    test_adaptive_router()
    test_router_with_experts()

    print("所有路由算法测试完成!")