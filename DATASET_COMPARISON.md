# 数据集处理方式对比

## 核心差异

### CulturalBench vs GlobalOpinions

| 方面 | CulturalBench | GlobalOpinions |
|------|---------------|----------------|
| **任务类型** | 二分类任务 | 概率分布预测任务 |
| **输入格式** | 问题 + 选项 | 问题 + 选项列表 |
| **输出格式** | 单一标签 (TRUE/FALSE) | 概率分布 [p1,p2,p3,p4,p5] |
| **模型输出** | 2维logits → softmax → 分类 | 5维logits → softmax → 概率分布 |
| **目标数据** | 离散标签 {0, 1} | 连续概率分布 |
| **损失函数** | 交叉熵损失 | JS散度损失 |
| **评估指标** | 准确率、F1分数 | JS相似度、概率分布距离 |

## 详细处理流程

### CulturalBench 处理流程

```python
# 1. 数据格式
{
  "prompt": "In Netherlands, what is cultural expectation?",
  "query": "Reaching destination exactly on time",
  "response": "TRUE"  # 离散标签
}

# 2. 数据处理
label = 1 if response == "TRUE" else 0  # 转换为数值标签

# 3. 模型前向传播
logits = model(input_ids, attention_mask)  # [batch_size, 2]
probabilities = F.softmax(logits, dim=-1)  # [batch_size, 2]
predictions = torch.argmax(logits, dim=-1)  # [batch_size]

# 4. 损失计算
loss = F.cross_entropy(logits, labels)

# 5. 评估指标
accuracy = (predictions == labels).float().mean()
```

### GlobalOpinions 处理流程

```python
# 1. 数据格式
{
  "prompt": "Please tell me your opinion of Coalition...",
  "query": "['Very favorable', 'Somewhat favorable', ...]",
  "response": "{'Greece': [0.01, 0.07, 0.3, 0.59, 0.03]}"  # 概率分布
}

# 2. 数据处理
target_probs = torch.tensor([0.01, 0.07, 0.3, 0.59, 0.03])  # 保持概率分布

# 3. 模型前向传播
logits = model(input_ids, attention_mask)  # [batch_size, 5]
pred_probs = F.softmax(logits, dim=-1)  # [batch_size, 5]
# 注意：这里没有argmax，保持概率分布形式

# 4. 损失计算
js_loss = js_divergence_loss(pred_probs, target_probs)

# 5. 评估指标
js_similarity = 1 - js_divergence(pred_probs, target_probs)
```

## 关键区别说明

### 1. 模型输出的含义不同

**CulturalBench**:
- 输出2维向量 `[logit_false, logit_true]`
- 经过softmax后得到 `[P(FALSE), P(TRUE)]`
- 最终预测：`argmax([P(FALSE), P(TRUE)])`

**GlobalOpinions**:
- 输出5维向量 `[logit1, logit2, logit3, logit4, logit5]`
- 经过softmax后得到 `[P1, P2, P3, P4, P5]`
- 最终预测：直接使用概率分布 `[P1, P2, P3, P4, P5]`，**不做argmax**

### 2. 评估方式不同

**CulturalBench**:
```python
# 离散比较
correct = (predictions == labels).float()
accuracy = correct.mean()
```

**GlobalOpinions**:
```python
# 连续分布比较
def js_divergence(p, q):
    m = (p + q) / 2
    return 0.5 * (kl_div(p, m) + kl_div(q, m))

similarity = 1 - js_divergence(pred_probs, target_probs)
```

### 3. 损失函数的数学原理

**交叉熵损失** (CulturalBench):
```
L_CE = -Σ y_i * log(p_i)
其中 y_i 是one-hot编码的真实标签
```

**JS散度损失** (GlobalOpinions):
```
L_JS = 1 - JS(P||Q)
其中 JS(P||Q) = 0.5 * [KL(P||M) + KL(Q||M)]
M = (P + Q) / 2
```

## 为什么不能混用？

1. **信息丢失**：将概率分布用argmax转换为单一标签会丢失分布信息
   - 例如：`[0.4, 0.3, 0.2, 0.1, 0.0]` 和 `[0.9, 0.05, 0.03, 0.01, 0.01]` 的argmax都是0，但分布含义完全不同

2. **评估失真**：用分类准确率评估概率分布任务无法反映模型的真实性能
   - 模型可能输出了很好的概率分布，但argmax错误就被判为失败

3. **优化目标不符**：JS散度优化的是整个分布的相似性，而不是单点的正确性

## 正确的实现方式

在我的代码修改中，虽然为了兼容现有框架结构，我保留了一些分类任务的接口（如pseudo_labels），但**核心评估指标是JS散度**：

```python
if self.model.dataset_config["dataset_type"] == "globalopinions":
    # 主要评估指标：JS散度
    js_metrics = compute_js_divergence_metrics(probabilities_array, target_probs_array)
    metrics = js_metrics  # 这是真正的评估结果
else:
    # 传统分类指标
    metrics = compute_metrics(predictions_array, labels_array)
```

所以回答您的问题：**GlobalOpinions数据集在损失计算和主要评估指标上并不是按分类任务处理的，而是按概率分布预测任务处理的**。我只是在代码兼容性方面保留了一些分类任务的结构。

