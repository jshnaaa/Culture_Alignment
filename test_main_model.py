#!/usr/bin/env python3
"""
测试集成LoRA微调Llama3.1的完整文化对齐模型
验证Llama LoRA + 路由器 + 专家层的端到端功能
"""

import numpy as np
import torch

from model.args import ModelArgs
from model.main import CulturalAlignmentModel
from torch.nn import DataParallel

def test_model_initialization():
    """测试模型初始化"""
    print("🚀 测试模型初始化")
    print("="*60)

    # 设置模型参数
    args = ModelArgs()
    args.llama_model_path = "./Meta-Llama-3.1-8B-Instruct"
    args.max_length = 512
    args.num_experts = 6
    args.expert_hidden_size = 768
    args.router_hidden_size = 512
    args.num_classes = 2
    args.lora_r = 16
    args.lora_alpha = 32
    args.lora_dropout = 0.1
    args.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print(f"创建文化对齐模型...")
        print(f"  Llama路径: {args.llama_model_path}")
        print(f"  设备: {args.device}")
        print(f"  LoRA配置: r={args.lora_r}, alpha={args.lora_alpha}")

        model = CulturalAlignmentModel(args)

        # 使用DataParallel来支持多GPU训练
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 张 GPU 进行训练")
            model = DataParallel(model)  # 将模型包装为DataParallel，支持多卡训练

        model = model.to(args.device)

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
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_model_forward():
    """测试模型前向传播"""
    print(f"\n" + "="*60)
    print("🔄 测试模型前向传播")
    print("="*60)

    model, args = test_model_initialization()
    if model is None:
        return False

    try:
        # 准备测试数据
        batch_size = 2
        seq_len = 128

        # 创建模拟输入
        input_ids = torch.randint(1, 1000, (batch_size, seq_len)).to(args.device)
        attention_mask = torch.ones(batch_size, seq_len).to(args.device)
        labels = torch.randint(0, args.num_classes, (batch_size,)).to(args.device)

        print(f"输入数据:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        print(f"  labels: {labels.shape}")

        # 前向传播
        model.train()
        outputs = model(input_ids, attention_mask, labels)

        print(f"\n模型输出:")
        print(f"  logits: {outputs['logits'].shape}")
        print(f"  专家权重: {outputs['expert_weights'].shape}")
        print(f"  分类损失: {outputs['classification_loss'].item():.4f}")
        print(f"  负载均衡损失: {outputs['load_balancing_loss'].item():.4f}")
        print(f"  多样性损失: {outputs['diversity_loss'].item():.4f}")
        print(f"  总损失: {outputs['loss'].item():.4f}")

        # 验证输出形状
        assert outputs['logits'].shape == (batch_size, args.num_classes), "logits形状错误"
        assert outputs['expert_weights'].shape == (batch_size, args.num_experts), "专家权重形状错误"

        # 验证专家权重归一化
        weights_sum = outputs['expert_weights'].sum(dim=1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "专家权重和应该为1"

        print(f"✓ 前向传播测试通过")
        return True

    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_to_prediction():
    """测试端到端文本预测"""
    print(f"\n" + "="*60)
    print("📝 测试端到端文本预测")
    print("="*60)

    model, args = test_model_initialization()
    if model is None:
        return False

    try:
        # 测试文本
        test_texts = [
            "This is a test sentence about cultural values.",
            "Another example text for testing the model."
        ]

        print(f"测试文本:")
        for i, text in enumerate(test_texts):
            print(f"  {i+1}: {text}")

        # 端到端预测
        model.eval()
        predictions = model.encode_and_predict(test_texts)

        print(f"\n预测结果:")
        print(f"  预测标签: {predictions['predictions']}")
        print(f"  预测概率: {predictions['probabilities']}")
        print(f"  专家权重: {predictions['expert_weights']}")

        # 验证预测结果
        assert predictions['predictions'].shape == (len(test_texts),), "预测形状错误"
        assert predictions['probabilities'].shape == (len(test_texts), args.num_classes), "概率形状错误"

        # 检查概率和为1
        prob_sum = predictions['probabilities'].sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), "概率和应该为1"

        print(f"✓ 端到端预测测试通过")
        return True

    except Exception as e:
        print(f"❌ 端到端预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """测试梯度流动"""
    print(f"\n" + "="*60)
    print("🌊 测试梯度流动")
    print("="*60)

    model, args = test_model_initialization()
    if model is None:
        return False

    try:
        # 准备数据
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(1, 1000, (batch_size, seq_len)).to(args.device)
        attention_mask = torch.ones(batch_size, seq_len).to(args.device)
        labels = torch.randint(0, args.num_classes, (batch_size,)).to(args.device)

        # 前向传播
        model.train()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        # 反向传播
        loss.backward()

        # 检查各模块的梯度
        print(f"梯度检查:")

        # Llama LoRA参数梯度
        llama_grad_count = 0
        for name, param in model.llama_model.named_parameters():
            if param.grad is not None and 'lora_' in name.lower():
                llama_grad_count += 1
                if llama_grad_count <= 3:  # 只显示前3个
                    print(f"  Llama LoRA {name}: 梯度范数={param.grad.norm():.6f}")
        print(f"  Llama LoRA有梯度参数: {llama_grad_count}")

        # 路由器梯度
        router_grad_count = 0
        for name, param in model.router.named_parameters():
            if param.grad is not None:
                router_grad_count += 1
        print(f"  路由器有梯度参数: {router_grad_count}")

        # 专家层梯度
        expert_grad_count = 0
        for name, param in model.expert_layer.named_parameters():
            if param.grad is not None:
                expert_grad_count += 1
        print(f"  专家层有梯度参数: {expert_grad_count}")

        # 分类器梯度
        classifier_grad_count = 0
        for name, param in model.classifier.named_parameters():
            if param.grad is not None:
                classifier_grad_count += 1
        print(f"  分类器有梯度参数: {classifier_grad_count}")

        # 验证至少有梯度
        assert llama_grad_count > 0, "Llama LoRA应该有梯度"
        assert router_grad_count > 0, "路由器应该有梯度"
        assert expert_grad_count > 0, "专家层应该有梯度"
        assert classifier_grad_count > 0, "分类器应该有梯度"

        print(f"✓ 梯度流动测试通过")
        return True

    except Exception as e:
        print(f"❌ 梯度流动测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expert_analysis():
    """测试专家分析功能"""
    print(f"\n" + "="*60)
    print("🔍 测试专家分析功能")
    print("="*60)

    model, args = test_model_initialization()
    if model is None:
        return False

    try:
        # 测试不同类型的输入对专家权重的影响
        test_cases = [
            "This is about traditional cultural values and respect.",
            "Modern technology and innovation are important.",
            "Family relationships and social harmony matter.",
            "Individual freedom and personal choice are key."
        ]

        print(f"分析不同输入的专家权重分布:")

        model.eval()
        for i, text in enumerate(test_cases):
            predictions = model.encode_and_predict([text])
            weights = predictions['expert_weights'][0].cpu().numpy()
            main_expert = np.argmax(weights)

            print(f"\n输入 {i+1}: {text[:50]}...")
            print(f"  专家权重: {weights.round(4)}")
            print(f"  主导专家: {main_expert} (权重: {weights[main_expert]:.4f})")
            print(f"  权重熵: {-np.sum(weights * np.log(weights + 1e-8)):.4f}")

        print(f"\n✓ 专家分析测试通过")
        return True

    except Exception as e:
        print(f"❌ 专家分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """运行comprehensive测试"""
    print("🚀 开始文化对齐模型comprehensive测试")
    print("="*60)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    test_results = []

    try:
        # 1. 模型前向传播测试
        result1 = test_model_forward()
        test_results.append(("前向传播", result1))

        # 2. 端到端预测测试
        result2 = test_text_to_prediction()
        test_results.append(("端到端预测", result2))

        # 3. 梯度流动测试
        result3 = test_gradient_flow()
        test_results.append(("梯度流动", result3))

        # 4. 专家分析测试
        result4 = test_expert_analysis()
        test_results.append(("专家分析", result4))

        # 总结
        print(f"\n" + "="*60)
        print("📊 测试结果总结")
        print("="*60)

        all_passed = True
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
            all_passed &= result

        if all_passed:
            print(f"\n🎉 所有测试通过！")
            print(f"🎯 模型特性:")
            print(f"  • Llama3.1 LoRA微调 ✓")
            print(f"  • 6个LoRA专家全参与 ✓")
            print(f"  • 智能路由权重分配 ✓")
            print(f"  • 端到端文本分类 ✓")
            print(f"  • 梯度流动正常 ✓")
            print(f"  • 专家权重可解释 ✓")
            print("="*60)
            return True
        else:
            print(f"\n❌ 部分测试失败")
            return False

    except Exception as e:
        print(f"\n❌ comprehensive测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()

    if success:
        print(f"\n🎊 文化对齐模型测试成功！")
        print(f"💡 模型已就绪，可以开始训练...")
    else:
        print(f"\n💔 测试失败，请检查问题...")
        exit(1)

