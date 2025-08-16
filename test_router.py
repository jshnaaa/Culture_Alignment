#!/usr/bin/env python3
"""
测试全专家参与的路由算法
验证所有6个专家都参与计算并进行加权求和
"""

import numpy as np
import torch

from model.experts import ExpertLayer
from model.router import ExpertRouter


def test_full_expert_participation():
    """测试全专家参与的计算"""
    print("="*60)
    print("测试全专家参与的路由算法")
    print("="*60)

    # 设置参数
    batch_size = 4
    input_dim = 768  # Llama hidden size
    num_experts = 6
    expert_hidden_dim = 512
    output_dim = 256

    # 创建路由器
    router = ExpertRouter(
        input_dim=input_dim,
        num_experts=num_experts,
        hidden_dim=512
    )

    # 创建专家层
    expert_layer = ExpertLayer(
        input_dim=input_dim,
        expert_hidden_dim=expert_hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts
    )

    # 创建模拟输入
    features = torch.randn(batch_size, input_dim)
    print(f"输入特征形状: {features.shape}")

    # 测试路由器
    print("\n--- 路由器测试 ---")
    expert_weights, router_logits = router(features)
    print(f"路由器输出权重形状: {expert_weights.shape}")
    print(f"专家权重和: {expert_weights.sum(dim=1)}")  # 应该全为1
    print(f"专家权重示例:")
    for i in range(min(2, batch_size)):
        weights = expert_weights[i].detach().numpy()
        print(f"  样本 {i}: {weights}")
        print(f"  权重和: {weights.sum():.6f}")

    # 验证权重和为1
    weights_sum = expert_weights.sum(dim=1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum)), "专家权重和应该为1"
    print("✓ 专家权重归一化验证通过")

    # 测试专家层
    print("\n--- 专家层测试 ---")
    weighted_output, expert_info = expert_layer(features, expert_weights)
    print(f"专家层最终输出形状: {weighted_output.shape}")

    # 调试：打印所有键
    print(f"expert_info中的所有键: {list(expert_info.keys())}")
    # 只匹配 expert_0, expert_1, ..., expert_5 这样的键
    import re
    expert_keys = [k for k in expert_info.keys() if re.match(r'^expert_\d+$', k)]
    print(f"专家输出键: {expert_keys}")
    print(f"专家个数: {len(expert_keys)}")

    # 验证所有专家都参与了计算
    individual_outputs = []
    for i in range(num_experts):
        expert_key = f"expert_{i}"
        if expert_key in expert_info:
            individual_outputs.append(expert_info[expert_key])
            print(f"专家 {i} 输出形状: {expert_info[expert_key].shape}")
        else:
            print(f"❌ 专家 {i} 的输出缺失！")

    print("✓ 所有专家都参与了计算")

    # 手动验证加权求和
    print("\n--- 加权求和验证 ---")
    manual_weighted_sum = torch.zeros_like(weighted_output)
    for i in range(num_experts):
        expert_contribution = expert_weights[:, i:i+1] * expert_info[f"expert_{i}"]
        manual_weighted_sum += expert_contribution

        # 显示每个专家的贡献
        contrib_norm = torch.norm(expert_contribution, dim=1).mean()
        print(f"专家 {i} 的平均贡献量: {contrib_norm:.4f}")

    # 验证手动计算与模型输出一致
    diff = torch.abs(weighted_output - manual_weighted_sum).max()
    print(f"手动计算与模型输出的最大差异: {diff:.8f}")
    assert diff < 1e-5, "手动计算与模型输出不一致"
    print("✓ 加权求和计算验证通过")

    # 测试专家权重的多样性
    print("\n--- 专家利用率分析 ---")
    utilization = router.get_expert_utilization(router_logits)
    print("专家利用率:")
    for i, util in enumerate(utilization):
        print(f"  专家 {i}: {util:.4f}")

    # 负载均衡损失
    load_balance_loss = router.compute_load_balancing_loss(router_logits)
    print(f"负载均衡损失: {load_balance_loss:.6f}")

    # 熵正则化
    entropy_loss = router.entropy_regularization(expert_weights)
    print(f"熵正则化损失: {entropy_loss:.6f}")

    print("\n" + "="*60)
    print("✅ 全专家参与路由算法测试通过！")
    print("🔹 所有6个专家都参与计算")
    print("🔹 权重通过softmax归一化（和为1）")
    print("🔹 最终输出为所有专家的加权求和")
    print("🔹 无top-k选择，真正的全专家参与")
    print("="*60)

def test_expert_weight_distribution():
    """测试专家权重分布的合理性"""
    print("\n" + "="*60)
    print("测试专家权重分布")
    print("="*60)

    input_dim = 768
    num_experts = 6
    router = ExpertRouter(input_dim=input_dim, num_experts=num_experts)

    # 创建多个不同的输入
    test_inputs = [
        torch.randn(1, input_dim),
        torch.zeros(1, input_dim),
        torch.ones(1, input_dim),
        torch.randn(1, input_dim) * 0.1,
        torch.randn(1, input_dim) * 2.0
    ]

    print("不同输入下的专家权重分布:")
    for i, test_input in enumerate(test_inputs):
        expert_weights, _ = router(test_input)
        weights = expert_weights[0].detach().numpy()

        print(f"\n输入 {i+1}:")
        print(f"  权重: {weights}")
        print(f"  最大权重专家: {np.argmax(weights)} (权重: {np.max(weights):.4f})")
        print(f"  权重标准差: {np.std(weights):.4f}")
        print(f"  权重熵: {-np.sum(weights * np.log(weights + 1e-8)):.4f}")

if __name__ == "__main__":
    # 设置随机种子以保证结果可重复
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        test_full_expert_participation()
        test_expert_weight_distribution()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

