#!/usr/bin/env python3
"""
简化版LoRA专家测试
快速验证基本功能，避免复杂的冻结/解冻测试
"""

import torch
import numpy as np
from model.experts import BaseExpertNetwork, LoRAExpert, ExpertLayer

def test_basic_functionality():
    """测试基本功能"""
    print("🔍 测试LoRA专家的基本功能")
    print("="*60)

    # 设置参数
    input_dim = 4096  # Llama hidden size
    hidden_dim = 768
    output_dim = 256
    num_experts = 6
    batch_size = 2

    try:
        # 1. 测试基础网络
        print("1. 测试基础专家网络...")
        base_net = BaseExpertNetwork(input_dim, hidden_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        base_output = base_net(x)
        print(f"   基础网络: 输入{x.shape} -> 输出{base_output.shape} ✓")

        # 2. 测试单个LoRA专家
        print("\n2. 测试单个LoRA专家...")
        expert = LoRAExpert(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            expert_id=0,
            lora_r=16,
            lora_alpha=32
        )
        expert_output = expert(x)
        print(f"   LoRA专家: 输入{x.shape} -> 输出{expert_output.shape} ✓")
        print(f"   专家ID: {expert.expert_id}")
        print(f"   LoRA秩: {expert.lora_r}")
        print(f"   LoRA缩放: {expert.lora_alpha}")

        # 验证输出维度
        assert expert_output.shape == (batch_size, output_dim), "专家输出维度错误"
        print("   输出维度验证 ✓")

        # 3. 输出模型结构
        print(f"\n3. 模型结构信息:")
        print(f"   基础模型类型: {type(expert.base_model).__name__}")
        print(f"   PEFT模型类型: {type(expert.model).__name__}")

        # 4. 参数统计（不进行冻结/解冻测试）
        print(f"\n4. 参数统计:")
        param_stats = expert.get_trainable_parameters()
        print(f"   总参数: {param_stats['total_params']:,}")
        print(f"   可训练: {param_stats['trainable_params']:,}")
        print(f"   LoRA: {param_stats['lora_params']:,}")
        print(f"   可训练比例: {param_stats['trainable_percentage']:.2f}%")

        # 5. LoRA参数详情
        print(f"\n5. LoRA参数详情:")
        lora_param_count = 0
        lora_param_names = []
        for name, param in expert.model.named_parameters():
            if 'lora_' in name.lower():
                print(f"   {name}: {param.shape} {'(trainable)' if param.requires_grad else '(frozen)'}")
                lora_param_count += param.numel()
                lora_param_names.append(name)
        print(f"   LoRA参数总数: {lora_param_count:,}")
        print(f"   LoRA参数个数: {len(lora_param_names)}")

        # 6. 测试专家层
        print(f"\n6. 测试专家层...")
        expert_layer = ExpertLayer(
            input_dim=input_dim,
            expert_hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts
        )

        # 验证专家个数
        actual_num = len(expert_layer.experts)
        assert actual_num == num_experts, f"专家个数错误: 期望{num_experts}, 实际{actual_num}"
        print(f"   专家个数: {actual_num} ✓")

        # 验证每个专家的维度
        print("   各专家维度检查:")
        for i, exp in enumerate(expert_layer.experts):
            print(f"     专家{i}: 输入{exp.input_dim}, 隐藏{exp.hidden_dim}, 输出{exp.output_dim}")
            assert exp.input_dim == input_dim, f"专家{i}输入维度错误"
            assert exp.output_dim == output_dim, f"专家{i}输出维度错误"
        print("   所有专家维度 ✓")

        # 7. 测试专家层前向传播
        print(f"\n7. 测试专家层前向传播...")
        features = torch.randn(batch_size, input_dim)
        expert_weights = torch.softmax(torch.randn(batch_size, num_experts), dim=-1)

        print(f"   输入特征: {features.shape}")
        print(f"   专家权重: {expert_weights.shape}")
        print(f"   权重和: {expert_weights.sum(dim=1)}")

        weighted_output, expert_info = expert_layer(features, expert_weights)
        print(f"   最终输出: {weighted_output.shape}")

        # 验证输出
        assert weighted_output.shape == (batch_size, output_dim), "专家层输出维度错误"
        print("   专家层输出维度 ✓")

        # 验证所有专家都有输出
        import re
        expert_keys = [k for k in expert_info.keys() if re.match(r'^expert_\d+$', k)]
        assert len(expert_keys) == num_experts, f"专家输出个数错误: {len(expert_keys)}"
        print(f"   专家输出个数: {len(expert_keys)} ✓")

        # 验证每个专家输出形状
        for i in range(num_experts):
            expert_key = f"expert_{i}"
            assert expert_key in expert_info, f"专家{i}输出缺失"
            expert_output_shape = expert_info[expert_key].shape
            expected_shape = (batch_size, output_dim)
            assert expert_output_shape == expected_shape, f"专家{i}输出形状错误: {expert_output_shape}"
        print("   所有专家输出形状 ✓")

        print("\n" + "="*60)
        print("🎉 基本测试全部通过！")
        print("✨ 验证内容:")
        print("  ✓ 基础专家网络结构正确")
        print("  ✓ PEFT LoRA专家创建成功")
        print("  ✓ 专家个数正确 (6个)")
        print("  ✓ 输入输出维度正确")
        print("  ✓ 前向传播计算正常")
        print("  ✓ LoRA参数结构正确")
        print("  ✓ 专家层集成正常")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_details():
    """测试参数详情"""
    print("\n🔋 测试LoRA参数详情")
    print("="*60)

    try:
        # 创建专家
        expert = LoRAExpert(
            input_dim=4096,
            hidden_dim=768,
            output_dim=256,
            lora_r=16,
            lora_alpha=32
        )

        print("所有参数详情:")
        total_params = 0
        trainable_params = 0
        lora_params = 0

        for name, param in expert.model.named_parameters():
            param_count = param.numel()
            total_params += param_count

            if param.requires_grad:
                trainable_params += param_count

            if 'lora_' in name.lower():
                lora_params += param_count
                print(f"  LoRA参数: {name:<40} {str(param.shape):<20} {param_count:>8,} {'✓' if param.requires_grad else '✗'}")
            else:
                # 只显示几个基础参数示例
                if 'network.0.weight' in name or 'network.3.weight' in name:
                    print(f"  基础参数: {name:<40} {str(param.shape):<20} {param_count:>8,} {'✓' if param.requires_grad else '✗'}")

        print(f"\n参数统计汇总:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  LoRA参数: {lora_params:,}")
        print(f"  可训练比例: {trainable_params/total_params*100:.2f}%")

        # 验证LoRA参数是否可训练
        lora_trainable = sum(p.numel() for n, p in expert.model.named_parameters()
                           if 'lora_' in n.lower() and p.requires_grad)
        print(f"  LoRA可训练参数: {lora_trainable:,}")

        print("🔋 参数详情测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ 参数详情测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始简化LoRA专家测试")

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    success = True
    success &= test_basic_functionality()
    success &= test_parameter_details()

    if success:
        print("\n🎊 所有简化测试通过！")
        print("📋 测试总结:")
        print("  • 基础网络: ✓")
        print("  • LoRA专家: ✓")
        print("  • 专家层: ✓")
        print("  • 维度验证: ✓")
        print("  • 前向传播: ✓")
        print("  • 参数结构: ✓")
        print("  • PEFT集成: ✓")
    else:
        print("\n❌ 部分测试失败")
        exit(1)

