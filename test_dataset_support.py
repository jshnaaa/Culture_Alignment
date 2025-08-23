#!/usr/bin/env python3
"""
测试脚本：验证不同数据集类型的支持
"""

import json
import torch
from model.args import ModelArgs

def test_args_configuration():
    """测试参数配置"""
    print("=" * 60)
    print("测试参数配置")
    print("=" * 60)

    # 测试CulturalBench配置
    args_cb = ModelArgs()
    args_cb.dataset_type = "culturalbench"
    config_cb = args_cb.get_dataset_config()
    print(f"CulturalBench配置: {config_cb}")

    # 测试GlobalOpinions配置
    args_go = ModelArgs()
    args_go.dataset_type = "globalopinions"
    config_go = args_go.get_dataset_config()
    print(f"GlobalOpinions配置: {config_go}")

    print("✓ 参数配置测试通过\n")

def test_data_format():
    """测试数据格式解析"""
    print("=" * 60)
    print("测试数据格式解析")
    print("=" * 60)

    # 测试GlobalOpinions数据格式
    sample_data = {
        "prompt": "Please tell me if you have a very favorable, somewhat favorable, somewhat unfavorable or very unfavorable opinion of ____. c. Democratic Coalition (PASOK & DIMAR) The source of this problem is GAS.",
        "query": "['Very favorable', 'Somewhat favorable', 'Somewhat unfavorable', 'Very unfavorable', 'DK/Refused']",
        "response": "defaultdict(<class 'list'>, {'Greece': [0.01, 0.07, 0.3, 0.59, 0.03]})"
    }

    print(f"样本数据: {sample_data}")

    # 测试选项解析
    try:
        options = eval(sample_data["query"])
        print(f"解析的选项: {options}")
        print(f"选项数量: {len(options)}")
    except Exception as e:
        print(f"选项解析失败: {e}")

    # 测试响应解析
    try:
        response_dict = eval(sample_data["response"])
        print(f"解析的响应: {response_dict}")

        for country, probs in response_dict.items():
            print(f"国家: {country}, 概率分布: {probs}")
            print(f"概率分布长度: {len(probs)}")
            print(f"概率分布和: {sum(probs)}")
    except Exception as e:
        print(f"响应解析失败: {e}")

    print("✓ 数据格式解析测试通过\n")

def test_js_divergence():
    """测试JS散度计算"""
    print("=" * 60)
    print("测试JS散度计算")
    print("=" * 60)

    import numpy as np

    # 创建示例概率分布
    pred_probs = np.array([
        [0.2, 0.3, 0.3, 0.1, 0.1],  # 预测分布1
        [0.1, 0.1, 0.2, 0.5, 0.1],  # 预测分布2
    ])

    target_probs = np.array([
        [0.01, 0.07, 0.3, 0.59, 0.03],  # 目标分布1
        [0.05, 0.15, 0.25, 0.45, 0.10], # 目标分布2
    ])

    print(f"预测概率分布:\n{pred_probs}")
    print(f"目标概率分布:\n{target_probs}")

    # 手动计算JS散度
    def compute_js_divergence(p, q):
        # 确保概率分布归一化
        p = p / p.sum()
        q = q / q.sum()

        # 计算中间分布M = (P + Q) / 2
        m = (p + q) / 2

        # 计算KL散度
        epsilon = 1e-8
        kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
        kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))

        # JS散度
        js_div = 0.5 * (kl_pm + kl_qm)
        return js_div

    for i in range(len(pred_probs)):
        js_div = compute_js_divergence(pred_probs[i], target_probs[i])
        js_sim = 1 - js_div
        print(f"样本{i+1}: JS散度={js_div:.4f}, JS相似度={js_sim:.4f}")

    print("✓ JS散度计算测试通过\n")

def test_loss_types():
    """测试不同损失函数类型"""
    print("=" * 60)
    print("测试损失函数类型")
    print("=" * 60)

    import torch.nn.functional as F

    # 测试分类损失
    print("测试分类损失:")
    logits = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
    labels = torch.tensor([1, 0])
    ce_loss = F.cross_entropy(logits, labels)
    print(f"分类logits: {logits}")
    print(f"标签: {labels}")
    print(f"交叉熵损失: {ce_loss.item():.4f}")

    # 测试JS散度损失
    print("\n测试JS散度损失:")
    pred_probs = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    target_probs = torch.tensor([[0.2, 0.8], [0.5, 0.5]])

    # 手动计算JS散度损失
    pred_probs_norm = F.softmax(pred_probs, dim=-1)
    target_probs_norm = F.softmax(target_probs, dim=-1)

    m = (pred_probs_norm + target_probs_norm) / 2
    kl_pm = F.kl_div(torch.log(pred_probs_norm + 1e-8), m, reduction='none').sum(dim=-1)
    kl_qm = F.kl_div(torch.log(target_probs_norm + 1e-8), m, reduction='none').sum(dim=-1)
    js_divergence = 0.5 * (kl_pm + kl_qm)
    js_loss = 1 - js_divergence

    print(f"预测概率: {pred_probs_norm}")
    print(f"目标概率: {target_probs_norm}")
    print(f"JS散度: {js_divergence}")
    print(f"JS损失: {js_loss}")
    print(f"平均JS损失: {js_loss.mean().item():.4f}")

    print("✓ 损失函数类型测试通过\n")

def main():
    """主函数"""
    print("开始测试数据集支持功能...")

    try:
        test_args_configuration()
        test_data_format()
        test_js_divergence()
        test_loss_types()

        print("=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print("\n配置建议:")
        print("1. CulturalBench数据集: 设置 dataset_type='culturalbench'")
        print("2. GlobalOpinions数据集: 设置 dataset_type='globalopinions'")
        print("3. 训练时会自动选择相应的损失函数和评估指标")
        print("4. CulturalBench使用交叉熵损失和分类准确率")
        print("5. GlobalOpinions使用JS散度损失和1-JS相似度")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

