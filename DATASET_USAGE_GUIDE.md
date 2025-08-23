# 数据集类型支持指南

本项目现已支持多种数据集类型，可以通过配置自动选择相应的损失函数和评估指标。

## 支持的数据集类型

### 1. CulturalBench数据集 (`culturalbench`)
- **数据格式**: 选择题格式，二分类任务
- **输入**: prompt（问题） + query（选项）
- **输出**: TRUE/FALSE
- **损失函数**: 交叉熵损失 (CrossEntropyLoss)
- **评估指标**: 准确率、精确率、召回率、F1分数

**示例数据**:
```json
{
  "prompt": "In the Netherlands, what is the cultural expectation regarding punctuality?",
  "query": "Reaching destination exactly on time",
  "response": "TRUE"
}
```

### 2. Global Opinions数据集 (`globalopinions`)
- **数据格式**: 概率预测任务
- **输入**: prompt（问题） + query（选项列表）
- **输出**: 各选项的概率分布
- **损失函数**: 1-JS散度损失
- **评估指标**: JS散度、JS相似度等概率分布相关指标

**示例数据**:
```json
{
  "prompt": "Please tell me if you have a very favorable, somewhat favorable, somewhat unfavorable or very unfavorable opinion of ____. c. Democratic Coalition (PASOK & DIMAR)",
  "query": "['Very favorable', 'Somewhat favorable', 'Somewhat unfavorable', 'Very unfavorable', 'DK/Refused']",
  "response": "defaultdict(<class 'list'>, {'Greece': [0.01, 0.07, 0.3, 0.59, 0.03]})"
}
```

### 3. CultureBank数据集 (`culturebank`)
- **状态**: 预留接口，待实现
- **配置**: 需要根据具体数据格式进行定制

## 配置方法

### 在 `args.py` 中设置数据集类型

```python
from model.args import ModelArgs

# CulturalBench数据集
args = ModelArgs()
args.dataset_type = "culturalbench"

# Global Opinions数据集
args = ModelArgs()
args.dataset_type = "globalopinions"
```

### 自动配置参数

设置数据集类型后，系统会自动配置：

- **类别数量** (`num_classes`)
- **损失函数类型** (`loss_type`)
- **输出类型** (`output_type`)

```python
# 获取当前数据集配置
config = args.get_dataset_config()
print(config)
# 输出示例:
# {
#     'dataset_type': 'globalopinions',
#     'num_classes': 5,
#     'loss_type': 'js_divergence',
#     'output_type': 'probability_distribution'
# }
```

## 代码修改要点

### 1. 主要修改文件

- **`model/args.py`**: 添加数据集类型配置和自动参数设置
- **`model/main.py`**: 添加JS散度损失函数和数据解析方法
- **`train.py`**: 修改数据集类和训练循环
- **`eval.py`**: 添加JS散度评估指标

### 2. 核心功能

#### JS散度损失函数
```python
def js_divergence_loss(self, pred_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """计算JS散度损失 (1 - JS散度)"""
    # 实现见 model/main.py
```

#### 数据集适配器
```python
class CulturalDataset(Dataset):
    """通用文化对齐数据集"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, dataset_type: str = "culturalbench"):
        # 根据数据集类型处理不同格式的数据
```

#### 评估指标计算
```python
def compute_js_divergence_metrics(pred_probs: np.ndarray, target_probs: np.ndarray) -> Dict[str, float]:
    """计算JS散度相关的评估指标"""
    # 实现见 eval.py
```

## 使用示例

### 训练CulturalBench数据集

```bash
python train.py \
    --dataset_type culturalbench \
    --train_data_path dataset_JSON/CulturalBench-Hard_train.json \
    --test_data_path dataset_JSON/CulturalBench-Hard_test.json \
    --epochs 10 \
    --batch_size 16
```

### 训练Global Opinions数据集

```bash
python train.py \
    --dataset_type globalopinions \
    --train_data_path dataset_JSON/global_opinions_train.json \
    --test_data_path dataset_JSON/global_opinions_test.json \
    --epochs 10 \
    --batch_size 16
```

### 模型评估

**重要说明**：模型评估已集成在训练脚本中，每个epoch结束后会自动评估。不需要单独运行eval.py脚本。

训练过程会自动：
- 根据数据集类型选择相应的评估指标
- CulturalBench：显示准确率、F1分数等分类指标
- GlobalOpinions：显示JS散度、JS相似度等概率分布指标

## 测试验证

运行测试脚本验证功能：

```bash
python test_dataset_support.py
```

测试内容包括：
- 参数配置正确性
- 数据格式解析
- JS散度计算
- 损失函数类型

## 添加新数据集类型

要添加新的数据集类型，需要：

1. **在 `args.py` 中添加配置**:
```python
def _configure_dataset_specific_params(self):
    # ... 现有代码 ...
    elif self.dataset_type == "new_dataset":
        self.num_classes = X  # 设置类别数
        self.loss_type = "new_loss_type"
        self.output_type = "new_output_type"
```

2. **在 `train.py` 中添加数据处理**:
```python
def _process_new_dataset_item(self, item):
    # 处理新数据集格式
    pass
```

3. **在 `main.py` 中添加损失函数**:
```python
elif self.dataset_config["loss_type"] == "new_loss_type":
    # 实现新的损失函数
    pass
```

4. **在 `eval.py` 中添加评估指标**:
```python
def compute_new_metrics(pred, target):
    # 实现新的评估指标
    pass
```

## 注意事项

1. **数据路径**: 确保数据文件路径正确
2. **格式检查**: 验证数据格式符合预期
3. **内存管理**: Global Opinions数据集可能更大，注意内存使用
4. **设备配置**: 确保GPU内存足够处理不同大小的数据集
5. **超参数调整**: 不同数据集可能需要不同的学习率和批次大小

通过这种设计，系统可以灵活支持多种数据集类型，并为未来的扩展提供了清晰的接口。

