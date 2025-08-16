#!/usr/bin/env python3
"""
测试PEFT LoRA专家实现
验证专家个数、模型结构、输入输出维度等
"""

import numpy as np
import torch

from model.experts import BaseExpertNetwork, LoRAExpert, ExpertLayer


def test_base_expert_network():
    """测试基础专家网络"""
    print("="*70)
    print("测试基础专家网络 (BaseExpertNetwork)")
    print("="*70)

    input_dim = 4096  # Llama hidden size
    hidden_dim = 512
    output_dim = 256

    # 创建基础网络
    base_network = BaseExpertNetwork(input_dim, hidden_dim, output_dim)

    print(f"基础网络结构:")
    print(base_network)

    # 测试输入输出
    batch_size = 2
    x = torch.randn(batch_size, input_dim)
    output = base_network(x)

    print(f"\n输入维度: {x.shape}")
    print(f"输出维度: {output.shape}")

    # 验证维度
    assert output.shape == (batch_size, output_dim), f"输出维度错误: {output.shape}"
    print("✓ 基础网络输入输出维度正确")

    # 统计参数
    total_params = sum(p.numel() for p in base_network.parameters())
    print(f"总参数数量: {total_params:,}")

    return base_network

def test_single_lora_expert():
    """测试单个LoRA专家"""
    print("\n" + "="*70)
    print("测试单个LoRA专家 (LoRAExpert)")
    print("="*70)

    input_dim = 4096
    hidden_dim = 768
    output_dim = 256
    expert_id = 0
    lora_r = 16
    lora_alpha = 32

    # 创建LoRA专家
    print(f"创建LoRA专家 {expert_id}...")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  LoRA秩: {lora_r}")
    print(f"  LoRA缩放: {lora_alpha}")

    lora_expert = LoRAExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        expert_id=expert_id,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    print(f"\n专家ID: {lora_expert.expert_id}")
    print(f"基础模型类型: {type(lora_expert.base_model).__name__}")
    print(f"PEFT模型类型: {type(lora_expert.model).__name__}")

    # 输出模型结构
    print(f"\n基础模型结构:")
    print(lora_expert.base_model)

    print(f"\nPEFT模型结构:")
    print(lora_expert.model)

    # 测试前向传播
    batch_size = 3
    x = torch.randn(batch_size, input_dim)
    output = lora_expert(x)

    print(f"\n前向传播测试:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")

    # 验证输出维度
    assert output.shape == (batch_size, output_dim), f"输出维度错误: {output.shape}"
    print("✓ LoRA专家输入输出维度正确")

    # 获取参数统计
    print(f"\n参数统计:")
    param_stats = lora_expert.get_trainable_parameters()
    print(f"  总参数: {param_stats['total_params']:,}")
    print(f"  可训练参数: {param_stats['trainable_params']:,}")
    print(f"  LoRA参数: {param_stats['lora_params']:,}")
    print(f"  可训练比例: {param_stats['trainable_percentage']:.2f}%")

    # 测试LoRA参数获取
    lora_params = lora_expert.get_lora_parameters()
    print(f"  LoRA参数个数: {len(lora_params)}")

    # 显示LoRA参数名称和形状
    print(f"\nLoRA参数详情:")
    for name, param in lora_expert.model.named_parameters():
        if 'lora_' in name.lower():
            print(f"  {name}: {param.shape} {'(trainable)' if param.requires_grad else '(frozen)'}")

    # 测试冻结/解冻功能
    print(f"\n测试冻结/解冻功能:")
    original_trainable = param_stats['trainable_params']

    # 冻结基础层
    lora_expert.freeze_base_layers()
    frozen_stats = lora_expert.get_trainable_parameters()
    print(f"  冻结后可训练参数: {frozen_stats['trainable_params']:,}")

    # 解冻基础层
    lora_expert.unfreeze_base_layers()
    unfrozen_stats = lora_expert.get_trainable_parameters()
    print(f"  解冻后可训练参数: {unfrozen_stats['trainable_params']:,}")

    assert unfrozen_stats['trainable_params'] == original_trainable, "解冻后参数数量不匹配"
    print("✓ 冻结/解冻功能正常")

    return lora_expert

def test_expert_layer():
    """测试专家层"""
    print("\n" + "="*70)
    print("测试专家层 (ExpertLayer)")
    print("="*70)

    input_dim = 4096
    expert_hidden_dim = 768
    output_dim = 256
    num_experts = 6
    lora_rank = 16
    lora_alpha = 32

    print(f"创建专家层...")
    print(f"  输入维度: {input_dim}")
    print(f"  专家隐藏维度: {expert_hidden_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  专家数量: {num_experts}")
    print(f"  LoRA秩: {lora_rank}")
    print(f"  LoRA缩放: {lora_alpha}")

    expert_layer = ExpertLayer(
        input_dim=input_dim,
        expert_hidden_dim=expert_hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha
    )

    print(f"\n专家层属性:")
    print(f"  专家数量: {expert_layer.num_experts}")
    print(f"  输入维度: {expert_layer.input_dim}")
    print(f"  专家隐藏维度: {expert_layer.expert_hidden_dim}")
    print(f"  输出维度: {expert_layer.output_dim}")
    print(f"  LoRA秩: {expert_layer.lora_rank}")
    print(f"  LoRA缩放: {expert_layer.lora_alpha}")

    # 验证专家个数
    actual_experts = len(expert_layer.experts)
    assert actual_experts == num_experts, f"专家个数错误: 期望{num_experts}, 实际{actual_experts}"
    print(f"✓ 专家个数正确: {actual_experts}")

    # 输出每个专家的信息
    print(f"\n各专家详细信息:")
    total_params_all = 0
    total_trainable_all = 0
    total_lora_all = 0

    for i, expert in enumerate(expert_layer.experts):
        print(f"\n--- 专家 {i} ---")
        print(f"  专家ID: {expert.expert_id}")
        print(f"  输入维度: {expert.input_dim}")
        print(f"  隐藏维度: {expert.hidden_dim}")
        print(f"  输出维度: {expert.output_dim}")
        print(f"  LoRA秩: {expert.lora_r}")
        print(f"  LoRA缩放: {expert.lora_alpha}")

        # 获取参数统计（不打印详情，避免输出过多）
        stats = expert.get_trainable_parameters()
        print(f"  参数统计:")
        print(f"    总参数: {stats['total_params']:,}")
        print(f"    可训练: {stats['trainable_params']:,}")
        print(f"    LoRA: {stats['lora_params']:,}")
        print(f"    可训练%: {stats['trainable_percentage']:.2f}%")

        total_params_all += stats['total_params']
        total_trainable_all += stats['trainable_params']
        total_lora_all += stats['lora_params']

        # 验证专家维度设置
        assert expert.input_dim == input_dim, f"专家{i}输入维度错误"
        assert expert.output_dim == output_dim, f"专家{i}输出维度错误"
        assert expert.lora_r == lora_rank, f"专家{i} LoRA秩错误"

    print(f"\n所有专家参数汇总:")
    print(f"  总参数: {total_params_all:,}")
    print(f"  可训练参数: {total_trainable_all:,}")
    print(f"  LoRA参数: {total_lora_all:,}")
    print(f"  平均可训练比例: {total_trainable_all/total_params_all*100:.2f}%")

    print("✓ 所有专家维度设置正确")

    return expert_layer

def test_expert_layer_forward():
    """测试专家层前向传播"""
    print("\n" + "="*70)
    print("测试专家层前向传播")
    print("="*70)

    input_dim = 4096
    expert_hidden_dim = 768
    output_dim = 256
    num_experts = 6
    batch_size = 4

    expert_layer = ExpertLayer(
        input_dim=input_dim,
        expert_hidden_dim=expert_hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts
    )

    # 创建输入
    features = torch.randn(batch_size, input_dim)
    expert_weights = torch.softmax(torch.randn(batch_size, num_experts), dim=-1)

    print(f"输入测试:")
    print(f"  特征形状: {features.shape}")
    print(f"  专家权重形状: {expert_weights.shape}")
    print(f"  专家权重和: {expert_weights.sum(dim=1)}")

    # 验证权重归一化
    weights_sum = expert_weights.sum(dim=1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum)), "专家权重和应该为1"
    print("✓ 专家权重归一化正确")

    # 前向传播
    print(f"\n执行前向传播...")
    weighted_output, expert_info = expert_layer(features, expert_weights)

    print(f"输出信息:")
    print(f"  加权输出形状: {weighted_output.shape}")
    print(f"  expert_info键: {list(expert_info.keys())}")

    # 验证输出维度
    expected_output_shape = (batch_size, output_dim)
    assert weighted_output.shape == expected_output_shape, f"输出形状错误: {weighted_output.shape}"
    print("✓ 输出形状正确")

    # 验证专家输出
    import re
    expert_output_keys = [k for k in expert_info.keys() if re.match(r'^expert_\d+$', k)]
    assert len(expert_output_keys) == num_experts, f"专家输出个数错误: {len(expert_output_keys)}"
    print(f"✓ 专家输出个数正确: {len(expert_output_keys)}")

    # 验证每个专家输出形状
    for i in range(num_experts):
        expert_key = f"expert_{i}"
        assert expert_key in expert_info, f"专家{i}输出缺失"
        expert_output_shape = expert_info[expert_key].shape
        expected_shape = (batch_size, output_dim)
        assert expert_output_shape == expected_shape, f"专家{i}输出形状错误: {expert_output_shape}"
        print(f"  专家{i}输出形状: {expert_output_shape} ✓")

    # 手动验证加权求和
    print(f"\n验证加权求和计算:")
    manual_sum = torch.zeros_like(weighted_output)
    for i in range(num_experts):
        contribution = expert_weights[:, i:i+1] * expert_info[f"expert_{i}"]
        manual_sum += contribution

        # 显示贡献
        contrib_norm = torch.norm(contribution, dim=1).mean()
        print(f"  专家{i}平均贡献: {contrib_norm:.4f}")

    # 验证计算一致性
    diff = torch.abs(weighted_output - manual_sum).max()
    print(f"  手动计算与模型输出最大差异: {diff:.8f}")
    assert diff < 1e-5, "手动计算与模型输出不一致"
    print("✓ 加权求和计算正确")

def test_single_expert_forward():
    """测试单个专家前向传播"""
    print("\n" + "="*70)
    print("测试单个专家前向传播")
    print("="*70)

    expert_layer = ExpertLayer(
        input_dim=4096,
        expert_hidden_dim=768,
        output_dim=256,
        num_experts=6
    )

    features = torch.randn(2, 4096)

    print(f"测试每个专家的单独前向传播:")
    for expert_id in range(expert_layer.num_experts):
        output = expert_layer.forward_single_expert(features, expert_id)
        print(f"  专家{expert_id}: 输入{features.shape} -> 输出{output.shape}")

        # 验证输出形状
        expected_shape = (2, 256)
        assert output.shape == expected_shape, f"专家{expert_id}输出形状错误"

    print("✓ 所有单个专家前向传播正确")

def test_expert_diversity():
    """测试专家多样性"""
    print("\n" + "="*70)
    print("测试专家多样性")
    print("="*70)

    expert_layer = ExpertLayer(
        input_dim=4096,
        expert_hidden_dim=768,
        output_dim=256,
        num_experts=6
    )

    features = torch.randn(4, 4096)

    # 计算专家多样性
    diversity_score = expert_layer.compute_expert_diversity(features)
    print(f"专家多样性分数: {diversity_score:.4f}")

    # 多样性应该在合理范围内
    assert 0 <= diversity_score <= 2, f"多样性分数异常: {diversity_score}"
    print("✓ 专家多样性计算正常")

    # 获取专家激活值
    activations = expert_layer.get_expert_activations(features)
    print(f"专家激活值键: {list(activations.keys())}")

    # 验证激活值
    for i in range(expert_layer.num_experts):
        key = f"expert_{i}_output"
        assert key in activations, f"专家{i}激活值缺失"
        activation_shape = activations[key].shape
        expected_shape = (4, 256)
        assert activation_shape == expected_shape, f"专家{i}激活值形状错误"

    print("✓ 专家激活值获取正常")

def run_comprehensive_test():
    """运行comprehensive测试"""
    print("🚀 开始LoRA专家comprehensive测试")
    print("="*70)

    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)

        # 1. 测试基础网络
        base_network = test_base_expert_network()

        # 2. 测试单个LoRA专家
        lora_expert = test_single_lora_expert()

        # 3. 测试专家层
        expert_layer = test_expert_layer()

        # 4. 测试专家层前向传播
        test_expert_layer_forward()

        # 5. 测试单个专家前向传播
        test_single_expert_forward()

        # 6. 测试专家多样性
        test_expert_diversity()

        print("\n" + "="*70)
        print("🎉 所有测试通过！")
        print("✨ 测试总结:")
        print("  🔸 基础专家网络 ✓")
        print("  🔸 PEFT LoRA专家 ✓")
        print("  🔸 专家个数验证 ✓")
        print("  🔸 模型结构输出 ✓")
        print("  🔸 输入输出维度 ✓")
        print("  🔸 前向传播计算 ✓")
        print("  🔸 参数统计分析 ✓")
        print("  🔸 冻结解冻功能 ✓")
        print("  🔸 专家多样性分析 ✓")
        print("="*70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_comprehensive_test()

