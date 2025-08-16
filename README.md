# 文化对齐分类模型

这个项目使用Llama 3.1模型进行文化问题的二分类任务，根据`prompt_question_country`预测答案是`TRUE`还是`FALSE`。

## 数据集格式

数据集应为JSON格式，每个样本包含以下字段：

```json
{
    "data_idx": "0",
    "question_idx": "0",
    "prompt_question": "In the Netherlands, which of the following is an unusual common public practice?",
    "prompt_option": "Cycle everywhere",
    "answer": "FALSE",
    "country": "Netherlands",
    "prompt_question_country": "In the Netherlands, which of the following is an unusual common public practice? The country corresponding to this question is Netherlands."
}
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## GPU 支持 🚀

本代码已完全优化支持GPU训练：
- **自动GPU检测**: 自动识别可用GPU并显示详细信息
- **混合精度训练**: 自动启用AMP，节省显存并加速训练
- **多GPU支持**: 自动分配模型到多个GPU
- **内存优化**: 使用float16精度，节省50%显存

**显存需求**：
- Llama 3.1-8B: ~8GB显存（训练时需要12-16GB）
- 建议配置：RTX 4080/4090 或同等GPU

详细GPU优化说明请参考：[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md)

## 使用方法

### 1. 准备数据

- 训练数据：`train_data.json`
- 验证数据（可选）：`val_data.json`

### 2. 修改模型路径

在`end2End.py`中修改`model_path`变量，指向您的Llama 3.1模型路径：

```python
model_path = "path/to/your/llama3.1/model"
```

### 3. 运行训练

```bash
python end2End.py
```

## 主要功能

### 数据加载
- `CultureAlignmentDataset`: 处理JSON格式数据集
- 使用`prompt_question_country`作为输入文本
- 将`answer`字段转换为二分类标签（TRUE→1, FALSE→0）

### 模型架构
- **预训练模型**: Llama 3.1 作为特征提取器
- **分类头**: 两层全连接网络，包含dropout和ReLU激活
- **池化策略**: 基于attention mask的加权平均池化

### 训练过程
- 使用AdamW优化器
- 交叉熵损失函数
- 支持验证集评估
- 实时显示训练损失和准确率

### 预测功能
- `predict_single_text()`: 对单个文本进行预测
- 返回预测结果（TRUE/FALSE）和置信度

## 输出文件

- `culture_alignment_model.pth`: 训练完成后保存的模型权重

## 配置参数

- `batch_size`: 批处理大小（默认8）
- `max_length`: 输入序列最大长度（默认512）
- `epochs`: 训练轮数（默认3）
- `learning_rate`: 学习率（默认1e-5）
- `hidden_dim`: 分类头隐藏层维度（默认256）
- `dropout_rate`: Dropout比例（默认0.1）

## 示例输出

```
Epoch 1/3:
  训练损失: 0.6234
  训练准确率: 0.7156
  验证准确率: 0.7342

示例预测:
输入: In the Netherlands, which of the following is an unusual common public practice? The country corresponding to this question is Netherlands.
预测: FALSE (置信度: 0.8234)

