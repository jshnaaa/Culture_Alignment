# 训练架构总结

## 核心改进

✅ **正确处理GlobalOpinions数据集**
- 不再将概率分布任务错误地当作分类任务处理
- 使用JS散度损失函数和相应的评估指标
- 保持概率分布的完整性，不做不必要的argmax操作

## 架构设计

### 1. 配置驱动 (`model/args.py`)
```python
# 根据数据集类型自动配置
dataset_type = "globalopinions"  # 或 "culturalbench"
# 自动设置：
# - num_classes (2 或 5)
# - loss_type ("classification" 或 "js_divergence")
# - output_type ("classification" 或 "probability_distribution")
```

### 2. 统一训练流程 (`train.py`)
```python
# 训练时自动选择损失函数
if dataset_config["loss_type"] == "js_divergence":
    loss = js_divergence_loss(pred_probs, target_probs)  # GlobalOpinions
else:
    loss = cross_entropy_loss(logits, labels)  # CulturalBench

# 评估时自动选择指标
if dataset_config["dataset_type"] == "globalopinions":
    metrics = compute_js_divergence_metrics(pred_probs, target_probs)
else:
    metrics = compute_classification_metrics(predictions, labels)
```

### 3. 数据集适配器 (`train.py`)
```python
class CulturalDataset:
    def __getitem__(self, idx):
        if self.dataset_type == "culturalbench":
            return self._process_culturalbench_item(item)  # 返回labels
        elif self.dataset_type == "globalopinions":
            return self._process_globalopinions_item(item)  # 返回target_probs
```

## 关键区别

| 方面 | CulturalBench | GlobalOpinions |
|------|---------------|----------------|
| **数据处理** | 离散标签 | 概率分布 |
| **模型输出** | 2维logits → argmax | 5维logits → softmax |
| **损失函数** | CrossEntropyLoss | JS散度损失 |
| **评估指标** | 准确率、F1分数 | JS相似度、概率分布距离 |
| **优化目标** | 单点预测正确性 | 整体分布相似性 |

## 主要评估指标

### CulturalBench指标
```python
{
    "accuracy": 0.85,
    "precision": 0.84,
    "recall": 0.86,
    "f1": 0.85,
    "loss": 0.32
}
```

### GlobalOpinions指标
```python
{
    "js_divergence": 0.12,      # JS散度 (越小越好)
    "js_similarity": 0.88,      # JS相似度 (越大越好)
    "js_std": 0.05,            # 标准差
    "loss": 0.15,
    # 参考指标
    "ref_accuracy": 0.75        # 仅作参考的分类准确率
}
```

## 使用方法

### 训练CulturalBench
```bash
python train.py --dataset_type culturalbench
```

### 训练GlobalOpinions
```bash
python train.py --dataset_type globalopinions
```

## ✅ 解决的问题

1. **任务混淆**：明确区分分类任务和概率分布预测任务
2. **信息丢失**：不再使用argmax丢失概率分布信息
3. **评估失真**：使用正确的评估指标衡量模型性能
4. **架构统一**：一套代码支持多种数据集类型

## 📝 注意事项

1. **评估集成**：评估功能已集成在训练脚本中，不需要单独的eval.py
2. **分布式支持**：在分布式训练中，GlobalOpinions暂时使用分类指标作为参考
3. **扩展性**：可以轻松添加新的数据集类型和评估指标
4. **兼容性**：保持与现有代码的兼容，同时支持新的概率分布任务

通过这种设计，系统现在能够正确处理概率分布预测任务，同时保持代码的简洁性和可扩展性。

