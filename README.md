# Cultural Alignment Model

基于 Llama3.1 的文化对齐任务模型，结合了 LoRA 微调、路由算法和专家混合（MoE）架构。

## 项目结构

```
Culture_Alignment/
├── data/                               # 数据目录
├── model/                              # 模型代码
│   ├── args.py                        # 参数配置
│   ├── llama.py                       # Llama3.1 共享层
│   ├── router.py                      # 路由算法
│   ├── experts.py                     # 专家层
│   └── main.py                        # 完整模型
├── train.py                           # 训练脚本
├── eval.py                            # 评估脚本
└── demo.py                            # 演示脚本
```

## 快速开始

### 1. 安装依赖
```bash
pip install torch transformers peft accelerate scikit-learn matplotlib seaborn tqdm
```

### 2. 训练模型
```bash
python train.py
```

### 3. 评估模型
```bash
python eval.py --model_path outputs/best_model
```

### 4. 演示使用
```bash
python demo.py --model_path outputs/best_model --mode interactive
```

## 模型架构

1. **Llama3.1 共享层**: LoRA 微调的特征提取器
2. **全专家路由算法**: 通过softmax输出所有6个专家的权重分布（和为1）
3. **专家层**: 6个独立的LoRA专家，**所有专家都参与计算**
4. **加权求和**: 使用路由权重对所有专家输出进行加权聚合
5. **分类头**: 二分类输出层

## 数据格式

```json
[
  {
    "prompt": "文化问题描述",
    "query": "选项或陈述",
    "response": "TRUE"
  }
]
```

## 主要特点

- **LoRA 高效微调**: 只训练少量参数，保持预训练知识
- **全专家参与架构**: 所有6个专家都参与计算，无top-k稀疏选择
- **智能路由算法**: softmax归一化确保专家权重和为1
- **加权求和输出**: 最终输出 = Σ(权重_i × 专家输出_i)
- **负载均衡和多样性正则化**: 促进专家特化和均衡使用
- **详细的评估分析**: 专家利用率、置信度分析等
- **交互式演示界面**: 支持单个预测和批量处理

## 测试验证

可以运行测试脚本验证路由算法的正确性：

```bash
python test_router.py
```

测试内容包括：
- ✅ 所有专家都参与计算
- ✅ 专家权重归一化（和为1）
- ✅ 加权求和计算正确性
- ✅ 专家利用率分析

