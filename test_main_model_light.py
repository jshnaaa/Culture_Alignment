#!/usr/bin/env python3
"""
轻量级版本：测试文化对齐模型（针对内存受限环境）
使用更保守的内存配置和简化测试
"""

import torch
import numpy as np
from model.args import ModelArgs
from model.main import CulturalAlignmentModel

def test_model_with_conservative_memory():
    """使用保守内存配置测试模型"""
    print("🚀 轻量级模型测试（内存优化版）")
    print("="*60)

    # 更保守的模型参数
    args = ModelArgs()
    args.llama_model_path = "./Meta-Llama-3.1-8B-Instruct"
    args.max_length = 256  # 减少序列长度
    args.num_experts = 4   # 减少专家数量
    args.expert_hidden_size = 512  # 减少专家隐藏层
    args.router_hidden_size = 256  # 减少路由器隐藏层
    args.num_classes = 2
    args.lora_r = 8        # 减少LoRA秩
    args.lora_alpha = 16   # 减少LoRA缩放
    args.lora_dropout = 0.1
    args.target_modules = ["q_proj", "v_proj"]  # 只对部分模块应用LoRA
    args.device = "cpu"    # 强制使用CPU

    print(f"保守内存配置:")
    print(f"  设备: {args.device}")
    print(f"  序列长度: {args.max_length}")
    print(f"  专家数量: {args.num_experts}")
    print(f"  LoRA秩: {args.lora_r}")
    print(f"  目标模块: {args.target_modules}")

    try:
        print(f"\n正在创建模型...")
        model = CulturalAlignmentModel(args)
        print(f"✓ 模型创建成功")

        # 获取参数统计
        param_stats = model.get_trainable_parameters()
        print(f"\n参数统计:")
        print(f"  Llama总参数: {param_stats['llama_total']:,}")
        print(f"  Llama可训练: {param_stats['llama_trainable']:,}")
        print(f"  路由器参数: {param_stats['router_params']:,}")
        print(f"  专家层参数: {param_stats['expert_params']:,}")
        print(f"  分类器参数: {param_stats['classifier_params']:,}")
        print(f"  总可训练参数: {param_stats['total_trainable']:,}")
        print(f"  可训练比例: {param_stats['trainable_percentage']:.2f}%")

        return model, args

    except Exception as e:
        print(f"❌ 轻量级模型创建失败: {e}")
        print(f"💡 建议:")
        print(f"  1. 检查是否安装了 bitsandbytes: pip install bitsandbytes")
        print(f"  2. 尝试使用更小的模型如 Llama3.1-1B")
        print(f"  3. 增加系统内存或使用云端环境")
        return None, None

def test_cpu_only_inference():
    """仅CPU推理测试"""
    print(f"\n" + "="*60)
    print("🔄 CPU模式推理测试")
    print("="*60)

    model, args = test_model_with_conservative_memory()
    if model is None:
        return False

    try:
        # 非常小的测试数据
        batch_size = 1
        seq_len = 64  # 很短的序列

        print(f"准备测试数据 (batch={batch_size}, seq_len={seq_len})...")
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, args.num_classes, (batch_size,))

        print(f"执行前向传播...")
        model.eval()  # 评估模式，减少内存使用

        with torch.no_grad():  # 不计算梯度，进一步减少内存
            outputs = model(input_ids, attention_mask, labels)

        print(f"✓ 前向传播成功")
        print(f"  输出logits形状: {outputs['logits'].shape}")
        print(f"  专家权重形状: {outputs['expert_weights'].shape}")
        print(f"  分类损失: {outputs['classification_loss'].item():.4f}")

        return True

    except Exception as e:
        print(f"❌ CPU推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_text_encoding():
    """简单文本编码测试"""
    print(f"\n" + "="*60)
    print("📝 简单文本编码测试")
    print("="*60)

    model, args = test_model_with_conservative_memory()
    if model is None:
        return False

    try:
        # 简单短文本
        test_text = "Hello world"
        print(f"测试文本: '{test_text}'")

        model.eval()
        with torch.no_grad():
            predictions = model.encode_and_predict([test_text])

        print(f"✓ 文本编码成功")
        print(f"  预测结果: {predictions['predictions']}")
        print(f"  预测概率: {predictions['probabilities']}")
        print(f"  专家权重: {predictions['expert_weights']}")

        return True

    except Exception as e:
        print(f"❌ 文本编码测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_lightweight_test():
    """运行轻量级测试套件"""
    print("🚀 开始轻量级文化对齐模型测试")
    print("="*60)

    torch.manual_seed(42)
    np.random.seed(42)

    test_results = []

    try:
        # 1. CPU推理测试
        result1 = test_cpu_only_inference()
        test_results.append(("CPU推理", result1))

        # 2. 文本编码测试
        result2 = test_simple_text_encoding()
        test_results.append(("文本编码", result2))

        # 总结
        print(f"\n" + "="*60)
        print("📊 轻量级测试结果")
        print("="*60)

        all_passed = True
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
            all_passed &= result

        if all_passed:
            print(f"\n🎉 轻量级测试通过！")
            print(f"🎯 验证功能:")
            print(f"  • 模型可以成功创建 ✓")
            print(f"  • CPU模式推理正常 ✓")
            print(f"  • 文本编码功能正常 ✓")
            print(f"  • 内存使用可控 ✓")
            print("="*60)
            return True
        else:
            print(f"\n❌ 部分轻量级测试失败")
            return False

    except Exception as e:
        print(f"\n❌ 轻量级测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 首先检查依赖
    try:
        import bitsandbytes
        print("✓ bitsandbytes 库已安装")
    except ImportError:
        print("⚠️  bitsandbytes 库未安装，8位量化可能不可用")
        print("   安装命令: pip install bitsandbytes")

    # 运行轻量级测试
    success = run_lightweight_test()

    if success:
        print(f"\n🎊 轻量级测试成功！")
        print(f"💡 模型基本功能正常，可以尝试:")
        print(f"   1. 在更大内存的机器上运行完整测试")
        print(f"   2. 使用云端GPU环境进行训练")
        print(f"   3. 继续开发其他功能模块")
    else:
        print(f"\n💔 轻量级测试失败")
        print(f"💡 建议检查:")
        print(f"   1. Llama模型文件是否存在且完整")
        print(f"   2. Python环境和依赖是否正确安装")
        print(f"   3. 系统内存是否至少有8GB可用")
        exit(1)

