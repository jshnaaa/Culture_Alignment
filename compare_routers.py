#!/usr/bin/env python3
"""
对比基础路由器和自适应路由器的区别
展示温度参数对专家权重分布的影响
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model.router import ExpertRouter, AdaptiveRouter

def compare_temperature_effects():
    """对比不同温度对专家权重分布的影响"""
    print("="*70)
    print("温度参数对专家权重分布的影响")
    print("="*70)

    # 创建基础路由器
    router = ExpertRouter(input_dim=768, num_experts=6)

    # 创建测试输入
    features = torch.randn(1, 768)
    _, router_logits = router(features)

    # 测试不同温度
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"原始logits: {router_logits[0].detach().numpy()}")
    print("\n不同温度下的专家权重分布：")
    print("-" * 70)
    print(f"{'温度':<8} {'专家权重分布':<50} {'熵值':<10}")
    print("-" * 70)

    for temp in temperatures:
        weights = torch.softmax(router_logits / temp, dim=-1)[0]
        weights_np = weights.detach().numpy()
        entropy = -np.sum(weights_np * np.log(weights_np + 1e-8))

        weights_str = " ".join([f"{w:.3f}" for w in weights_np])
        print(f"{temp:<8} [{weights_str}] {entropy:<10.3f}")

    print("-" * 70)
    print("📊 观察：")
    print("  • 低温度（0.1）：权重集中，熵值小，专家选择更确定")
    print("  • 高温度（5.0）：权重均匀，熵值大，专家选择更平均")

def compare_routers():
    """对比基础路由器和自适应路由器"""
    print("\n" + "="*70)
    print("基础路由器 vs 自适应路由器")
    print("="*70)

    # 创建两种路由器
    basic_router = ExpertRouter(input_dim=768, num_experts=6)
    adaptive_router = AdaptiveRouter(input_dim=768, num_experts=6)

    # 创建不同类型的测试输入
    test_cases = [
        ("随机输入", torch.randn(1, 768)),
        ("零向量", torch.zeros(1, 768)),
        ("单位向量", torch.ones(1, 768)),
        ("小幅随机", torch.randn(1, 768) * 0.1),
        ("大幅随机", torch.randn(1, 768) * 3.0),
    ]

    print(f"{'输入类型':<12} {'基础路由器':<35} {'自适应路由器':<35} {'温度':<8}")
    print("-" * 95)

    for case_name, features in test_cases:
        # 基础路由器（固定温度=1.0）
        basic_weights, _ = basic_router(features)
        basic_weights_np = basic_weights[0].detach().numpy()
        basic_entropy = -np.sum(basic_weights_np * np.log(basic_weights_np + 1e-8))

        # 自适应路由器（动态温度）
        adaptive_weights, _, temp = adaptive_router(features)
        adaptive_weights_np = adaptive_weights[0].detach().numpy()
        adaptive_entropy = -np.sum(adaptive_weights_np * np.log(adaptive_weights_np + 1e-8))
        adaptive_temp = temp[0].item()

        # 格式化权重显示
        basic_str = f"[{' '.join([f'{w:.2f}' for w in basic_weights_np[:3]])}...] η={basic_entropy:.2f}"
        adaptive_str = f"[{' '.join([f'{w:.2f}' for w in adaptive_weights_np[:3]])}...] η={adaptive_entropy:.2f}"

        print(f"{case_name:<12} {basic_str:<35} {adaptive_str:<35} {adaptive_temp:.2f}")

    print("-" * 95)
    print("📈 分析：")
    print("  • 基础路由器：所有输入都使用固定温度（1.0）")
    print("  • 自适应路由器：根据输入特征动态调整温度（0.1-2.0）")
    print("  • η 表示熵值，越高表示权重分布越均匀")

def visualize_weight_distributions():
    """可视化专家权重分布"""
    print("\n" + "="*70)
    print("专家权重分布可视化")
    print("="*70)

    # 创建路由器
    basic_router = ExpertRouter(input_dim=768, num_experts=6)
    adaptive_router = AdaptiveRouter(input_dim=768, num_experts=6)

    # 生成多个测试样本
    num_samples = 20
    features_list = [torch.randn(1, 768) for _ in range(num_samples)]

    basic_weights_all = []
    adaptive_weights_all = []
    temperatures = []

    for features in features_list:
        # 基础路由器
        basic_weights, _ = basic_router(features)
        basic_weights_all.append(basic_weights[0].detach().numpy())

        # 自适应路由器
        adaptive_weights, _, temp = adaptive_router(features)
        adaptive_weights_all.append(adaptive_weights[0].detach().numpy())
        temperatures.append(temp[0].item())

    # 计算统计信息
    basic_weights_array = np.array(basic_weights_all)
    adaptive_weights_array = np.array(adaptive_weights_all)

    # 计算每个专家的平均利用率和标准差
    print("专家利用率统计：")
    print(f"{'专家ID':<8} {'基础路由器':<20} {'自适应路由器':<20}")
    print("-" * 50)

    for i in range(6):
        basic_mean = basic_weights_array[:, i].mean()
        basic_std = basic_weights_array[:, i].std()
        adaptive_mean = adaptive_weights_array[:, i].mean()
        adaptive_std = adaptive_weights_array[:, i].std()

        print(f"专家 {i:<3} {basic_mean:.3f}±{basic_std:.3f}     {adaptive_mean:.3f}±{adaptive_std:.3f}")

    # 温度统计
    print(f"\n自适应温度统计：")
    print(f"  平均温度: {np.mean(temperatures):.3f}")
    print(f"  温度范围: {np.min(temperatures):.3f} - {np.max(temperatures):.3f}")
    print(f"  温度标准差: {np.std(temperatures):.3f}")

def demonstrate_adaptation_benefits():
    """演示自适应路由的优势"""
    print("\n" + "="*70)
    print("自适应路由的优势演示")
    print("="*70)

    adaptive_router = AdaptiveRouter(input_dim=768, num_experts=6)

    # 创建不同"难度"的输入
    test_scenarios = [
        ("简单模式", torch.randn(1, 768) * 0.5),    # 小方差，可能需要更集中的专家
        ("复杂模式", torch.randn(1, 768) * 2.0),    # 大方差，可能需要更多专家参与
        ("标准模式", torch.randn(1, 768)),           # 标准方差
    ]

    print("不同场景下的自适应行为：")
    print(f"{'场景':<12} {'温度':<8} {'主导专家权重':<12} {'权重熵':<10} {'策略'}")
    print("-" * 60)

    for scenario_name, features in test_scenarios:
        weights, logits, temp = adaptive_router(features)
        weights_np = weights[0].detach().numpy()
        temp_val = temp[0].item()

        max_weight = np.max(weights_np)
        entropy = -np.sum(weights_np * np.log(weights_np + 1e-8))

        if temp_val < 0.8:
            strategy = "专家特化"
        elif temp_val > 1.2:
            strategy = "专家协作"
        else:
            strategy = "均衡模式"

        print(f"{scenario_name:<12} {temp_val:<8.3f} {max_weight:<12.3f} {entropy:<10.3f} {strategy}")

    print("-" * 60)
    print("🎯 自适应优势：")
    print("  • 根据输入复杂度动态调整专家参与度")
    print("  • 简单任务：集中专家资源（低温度）")
    print("  • 复杂任务：广泛专家协作（高温度）")
    print("  • 提高模型的表达能力和适应性")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        compare_temperature_effects()
        compare_routers()
        visualize_weight_distributions()
        demonstrate_adaptation_benefits()

        print("\n" + "="*70)
        print("🎉 路由器对比分析完成！")
        print("💡 建议：根据任务复杂度选择合适的路由器")
        print("  • 简单任务：ExpertRouter（计算效率高）")
        print("  • 复杂任务：AdaptiveRouter（适应性强）")
        print("="*70)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

