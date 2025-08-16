import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    from torch.amp import autocast, GradScaler

# GPU优化设置
torch.backends.cudnn.benchmark = True  # 加速卷积运算
torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32精度
torch.backends.cudnn.allow_tf32 = True

# GPU检查和内存管理函数
def check_gpu_availability():
    """检查GPU可用性并显示信息"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU设备")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # 清空GPU缓存
        torch.cuda.empty_cache()
        return True
    else:
        print("❌ 未检测到可用的GPU，将使用CPU训练")
        return False

def get_device():
    """获取最佳设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 使用GPU设备: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("🔄 使用CPU设备")
    return device

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# 1. 数据集类定义
class CultureAlignmentDataset(Dataset):
    def __init__(self, json_file_path, tokenizer, max_length=512):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 使用prompt_question_country作为输入
        text = item['prompt'] + item['query']
        # 使用answer作为标签，TRUE->1, FALSE->0
        label = 1 if item['response'] == 'TRUE' else 0

        # 文本编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 2. 加载Llama 3.1模型和tokenizer
model_path = "Meta-Llama-3.1-8B-Instruct"

# 检查GPU可用性
device = get_device()
gpu_available = check_gpu_availability()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型，优化GPU内存使用
print("正在加载Llama 3.1模型...")
if gpu_available:
    # GPU模式：使用torch_dtype和device_map优化
    llama_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用half精度节省显存
        device_map="auto",  # 自动分配到多GPU
        trust_remote_code=True
    )
else:
    # CPU模式
    llama_model = AutoModel.from_pretrained(model_path)

print(f" 模型加载完成，参数量: {sum(p.numel() for p in llama_model.parameters()) / 1e9:.1f}B")

# 3. 定义分类头模型（二分类）
class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 二分类：TRUE/FALSE

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. 完整的模型类
class CultureAlignmentModel(nn.Module):
    def __init__(self, llama_model, hidden_size):
        super().__init__()
        self.llama_model = llama_model
        self.classification_head = BinaryClassificationHead(hidden_size)

    def forward(self, input_ids, attention_mask):
        # 获取Llama模型输出
        outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 使用最后一层隐藏状态
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # 对序列进行池化（使用CLS token或平均池化）
        # 这里使用平均池化，只对有效token进行平均
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask

        # 通过分类头
        logits = self.classification_head(sentence_embedding)
        return logits

# 5. 优化训练函数（支持混合精度和GPU优化）
def train_model(model, train_loader, val_loader, epochs=3, lr=1e-5, use_amp=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 如果模型没有被device_map自动放置，则手动移动
    if not hasattr(model.llama_model, 'hf_device_map'):
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 混合精度训练
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    print(f"🎯 开始训练 - 设备: {device}")
    if scaler:
        print("📊 启用混合精度训练（AMP）")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        for i, batch in enumerate(train_loader):
            # 数据移动到GPU
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # 使用混合精度
            if scaler:
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # 收集预测结果
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

            # 定期显示进度
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

                # 清理GPU内存
                if torch.cuda.is_available() and (i + 1) % 50 == 0:
                    clear_gpu_memory()

        # 计算训练准确率
        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)

        print(f' Epoch {epoch+1}/{epochs} 完成:')
        print(f'   训练损失: {avg_loss:.4f}')
        print(f'   训练准确率: {train_acc:.4f}')

        # 验证阶段
        if val_loader:
            val_acc = evaluate_model(model, val_loader, device)
            print(f'  ✔️  验证准确率: {val_acc:.4f}')

        # GPU内存状态
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f'  🖥️  GPU内存: {memory_allocated:.1f}GB / {memory_reserved:.1f}GB')

        print()

# 6. 评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# 7. 预测函数
def predict_single_text(model, tokenizer, text, device, max_length=512):
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        confidence = probs[0][pred].item()

    result = 'TRUE' if pred == 1 else 'FALSE'
    return result, confidence

# 8. 主执行部分
if __name__ == "__main__":
    # 数据路径（需要根据实际路径修改）
    train_data_path = "data/CulturalBench-Hard_train.json"  # 训练数据路径
    val_data_path = "data/CulturalBench-Hard_test.json"  # 验证数据路径（可选）

    # 创建数据集
    print("正在加载数据集...")
    train_dataset = CultureAlignmentDataset(train_data_path, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 验证数据（如果存在）
    val_loader = None
    try:
        val_dataset = CultureAlignmentDataset(val_data_path, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    except FileNotFoundError:
        print("未找到验证数据，将只使用训练数据")

    # 创建模型
    print("正在初始化模型...")
    # 假设Llama 3.1的hidden_size为4096，需要根据实际模型调整
    hidden_size = llama_model.config.hidden_size
    model = CultureAlignmentModel(llama_model, hidden_size)

    # 训练模型
    print("开始训练...")
    train_model(model, train_loader, val_loader, epochs=3, lr=1e-5)

    # 保存模型
    torch.save(model.state_dict(), 'culture_alignment_model.pth')
    print("模型已保存到 culture_alignment_model.pth")

    # 示例预测
    print("\n示例预测:")

    sample_text = "In the Netherlands, which of the following is an unusual common public practice? The country corresponding to this question is Netherlands."
    result, confidence = predict_single_text(model, tokenizer, sample_text, device)
    print(f"输入: {sample_text}")
    print(f"预测: {result} (置信度: {confidence:.4f})")

    # 清理GPU内存
    clear_gpu_memory()
    print("🧹 已清理GPU内存")
