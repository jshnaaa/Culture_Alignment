import json
import logging
import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from model.args import ModelArgs
from model.main import CulturalAlignmentModel
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_distributed_backend(args):
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )
        
        torch.cuda.set_device(args.local_rank)
        args.distributed = True
        args.device = torch.device(f"cuda:{args.local_rank}")
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = -1
        args.distributed = False
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return args.device

class CulturalBenchDataset(Dataset):
    """文化对齐数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 提取数据
        prompt = item["prompt"]
        query = item["query"]
        response = item["response"]

        # 构建输入文本
        input_text = f"Question: {prompt}\nOption: {query}\nAnswer:"

        # 标签转换：TRUE -> 1, FALSE -> 0
        label = 1 if response == "TRUE" else 0

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
            "prompt": prompt,
            "query": query,
            "response": response
        }

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, distributed: bool = False) -> DataLoader:
    """创建数据加载器"""
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle
        )
        shuffle = False  # DistributedSampler已经处理了shuffle
    else:
        sampler = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # 在分布式训练中，确保每个GPU的批次大小一致
    )

def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, save_path: str):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FALSE', 'TRUE'],
                yticklabels=['FALSE', 'TRUE'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model: CulturalAlignmentModel,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: str,
                gradient_accumulation_steps: int = 1,
                distributed: bool = False) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_classification_loss = 0
    total_load_balancing_loss = 0
    total_diversity_loss = 0
    num_batches = 0

    # 只在主进程中显示进度条
    if not distributed or dist.get_rank() == 0:
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader

    for step, batch in enumerate(progress_bar):
        # 移动到设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask, labels)

        # 获取损失
        loss = outputs["loss"]
        classification_loss = outputs["classification_loss"]
        load_balancing_loss = outputs["load_balancing_loss"]
        diversity_loss = outputs["diversity_loss"]

        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # 更新参数
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 累积损失
        total_loss += loss.item() * gradient_accumulation_steps
        total_classification_loss += classification_loss.item()
        total_load_balancing_loss += load_balancing_loss.item()
        total_diversity_loss += diversity_loss.item()
        num_batches += 1

        # 更新进度条
        if not distributed or dist.get_rank() == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

    # 在分布式环境中，计算所有进程的平均损失
    if distributed:
        # 将所有进程的损失收集到主进程
        losses = torch.tensor([total_loss, total_classification_loss, 
                              total_load_balancing_loss, total_diversity_loss, num_batches], 
                             dtype=torch.float32, device=device)
        
        # 收集所有进程的损失
        dist.reduce(losses, dst=0, op=dist.ReduceOp.SUM)
        
        if dist.get_rank() == 0:
            total_loss, total_classification_loss, total_load_balancing_loss, \
            total_diversity_loss, total_num_batches = losses.tolist()

            # 防止除零错误
            if total_num_batches > 0:
                return {
                    "total_loss": total_loss / total_num_batches,
                    "classification_loss": total_classification_loss / total_num_batches,
                    "load_balancing_loss": total_load_balancing_loss / total_num_batches,
                    "diversity_loss": total_diversity_loss / total_num_batches
                }
            else:
                logger.warning("Total number of batches is zero!")
                return {
                    "total_loss": 0.0,  # 默认值
                    "classification_loss": 0.0,
                    "load_balancing_loss": 0.0,
                    "diversity_loss": 0.0
                }
        else:
            return {
                "total_loss": 0.0,  # 默认值
                "classification_loss": 0.0,
                "load_balancing_loss": 0.0,
                "diversity_loss": 0.0
            }
    else:
        if num_batches > 0:
            return {
                "total_loss": total_loss / num_batches,
                "classification_loss": total_classification_loss / num_batches,
                "load_balancing_loss": total_load_balancing_loss / num_batches,
                "diversity_loss": total_diversity_loss / num_batches
            }
        else:
            logger.warning("Total number of batches is zero!")
            return {
                "total_loss": 0.0,  # 默认值
                "classification_loss": 0.0,
                "load_balancing_loss": 0.0,
                "diversity_loss": 0.0
            }

# def train_epoch(model: CulturalAlignmentModel,
#                 dataloader: DataLoader,
#                 optimizer: torch.optim.Optimizer,
#                 scheduler: torch.optim.lr_scheduler._LRScheduler,
#                 device: str,
#                 gradient_accumulation_steps: int = 1,
#                 distributed: bool = False) -> Dict[str, float]:
#     """训练一个epoch"""
#     model.train()
#     total_loss = 0
#     total_classification_loss = 0
#     total_load_balancing_loss = 0
#     total_diversity_loss = 0
#     num_batches = 0

#     # 只在主进程中显示进度条
#     if not distributed or dist.get_rank() == 0:
#         progress_bar = tqdm(dataloader, desc="Training")
#     else:
#         progress_bar = dataloader

#     for step, batch in enumerate(progress_bar):
#         # 移动到设备
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         # 前向传播
#         outputs = model(input_ids, attention_mask, labels)

#         # 获取损失
#         loss = outputs["loss"]
#         classification_loss = outputs["classification_loss"]
#         load_balancing_loss = outputs["load_balancing_loss"]
#         diversity_loss = outputs["diversity_loss"]

#         # 梯度累积
#         loss = loss / gradient_accumulation_steps
#         loss.backward()

#         # 更新参数
#         if (step + 1) % gradient_accumulation_steps == 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()

#         # 累积损失
#         total_loss += loss.item() * gradient_accumulation_steps
#         total_classification_loss += classification_loss.item()
#         total_load_balancing_loss += load_balancing_loss.item()
#         total_diversity_loss += diversity_loss.item()
#         num_batches += 1

#         # 只在主进程中更新进度条
#         if not distributed or dist.get_rank() == 0:
#             progress_bar.set_postfix({
#                 'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
#                 'lr': f'{scheduler.get_last_lr()[0]:.2e}'
#             })

#     # 在分布式环境中，计算所有进程的平均损失
#     if distributed:
#         # 将所有进程的损失收集到主进程
#         losses = torch.tensor([total_loss, total_classification_loss, 
#                               total_load_balancing_loss, total_diversity_loss, num_batches], 
#                              dtype=torch.float32, device=device)
        
#         # 收集所有进程的损失
#         dist.reduce(losses, dst=0, op=dist.ReduceOp.SUM)
        
#         if dist.get_rank() == 0:
#             total_loss, total_classification_loss, total_load_balancing_loss, \
#             total_diversity_loss, total_num_batches = losses.tolist()
            
#             return {
#                 "total_loss": total_loss / total_num_batches,
#                 "classification_loss": total_classification_loss / total_num_batches,
#                 "load_balancing_loss": total_load_balancing_loss / total_num_batches,
#                 "diversity_loss": total_diversity_loss / total_num_batches
#             }
#         else:
#             return {}
#     else:
#         return {
#             "total_loss": total_loss / num_batches,
#             "classification_loss": total_classification_loss / num_batches,
#             "load_balancing_loss": total_load_balancing_loss / num_batches,
#             "diversity_loss": total_diversity_loss / num_batches
#         }

def evaluate(model: CulturalAlignmentModel,
             dataloader: DataLoader,
             device: str,
             distributed: bool = False) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        # 只在主进程中显示进度条
        if not distributed or dist.get_rank() == 0:
            progress_bar = tqdm(dataloader, desc="Evaluating")
        else:
            progress_bar = dataloader

        for batch in progress_bar:
            # 移动到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask, labels)

            # 获取预测
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)

            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += outputs["loss"].item()
            num_batches += 1

    # 在分布式环境中，收集所有进程的结果
    if distributed:
        # 将所有预测和标签收集到主进程
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # 收集所有进程的预测和标签
        predictions_tensor = torch.tensor(all_predictions, device=device)
        labels_tensor = torch.tensor(all_labels, device=device)
        
        # 收集所有进程的结果
        gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_predictions, predictions_tensor)
        dist.all_gather(gathered_labels, labels_tensor)
        
        # 收集损失
        losses = torch.tensor([total_loss, num_batches], dtype=torch.float32, device=device)
        dist.reduce(losses, dst=0, op=dist.ReduceOp.SUM)
        
        if dist.get_rank() == 0:
            # 合并所有进程的结果
            all_predictions = torch.cat(gathered_predictions).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()
            total_loss, total_num_batches = losses.tolist()
            
            # 计算指标
            metrics = compute_metrics(all_predictions, all_labels)
            metrics["loss"] = total_loss / total_num_batches
            
            return metrics, all_predictions, all_labels
        else:
            return {}, [], []
    else:
        # 计算指标
        metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
        metrics["loss"] = total_loss / num_batches
        
        return metrics, all_predictions, all_labels

# def main():
#     # 解析参数
#     args = ModelArgs()

#     # 初始化分布式环境
#     device = init_distributed_backend(args)

#     # 设置随机种子
#     set_seed(args.seed)

#     # 只在主进程中创建输出目录和设置日志
#     if not args.distributed or dist.get_rank() == 0:
#         # 创建输出目录
#         os.makedirs(args.output_dir, exist_ok=True)

#         # 设置日志文件
#         file_handler = logging.FileHandler(args.log_file)
#         file_handler.setLevel(logging.INFO)
#         logger.addHandler(file_handler)

#         logger.info("Starting Cultural Alignment Model Training")
#         logger.info(f"Arguments: {args}")

#     # 初始化模型
#     if not args.distributed or dist.get_rank() == 0:
#         logger.info("Initializing model...")
    
#     model = CulturalAlignmentModel(args)
#     model.to(device)

#     # 在分布式环境中使用DDP包装模型之前，先保存tokenizer
#     tokenizer = model.llama_shared.tokenizer  # 先获取tokenizer

#     # 在分布式环境中使用DDP包装模型
#     if args.distributed:
#         model = torch.nn.parallel.DistributedDataParallel(
#             model,
#             device_ids=[args.local_rank],
#             output_device=args.local_rank
#         )

#     # 创建数据集 - 使用之前保存的tokenizer
#     if not args.distributed or dist.get_rank() == 0:
#         logger.info("Loading datasets...")
    
#     train_dataset = CulturalBenchDataset(
#         args.train_data_path,
#         tokenizer,  # 使用保存的tokenizer而不是model.llama_shared.tokenizer
#         args.max_length
#     )

#     test_dataset = CulturalBenchDataset(
#         args.test_data_path,
#         tokenizer,  # 使用保存的tokenizer而不是model.llama_shared.tokenizer
#         args.max_length
#     )

#     # 创建数据加载器
#     train_dataloader = create_dataloader(
#         train_dataset, 
#         args.batch_size, 
#         shuffle=True, 
#         distributed=args.distributed
#     )
    
#     test_dataloader = create_dataloader(
#         test_dataset, 
#         args.batch_size, 
#         shuffle=False, 
#         distributed=args.distributed
#     )

#     # 设置优化器
#     optimizer = AdamW(
#         model.parameters(),
#         lr=args.learning_rate,
#         weight_decay=args.weight_decay
#     )

#     # 计算总步数
#     total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

#     # 设置学习率调度器
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=args.warmup_steps,
#         num_training_steps=total_steps
#     )

#     # 训练循环
#     best_accuracy = 0
#     train_losses = []
#     eval_accuracies = []

#     if not args.distributed or dist.get_rank() == 0:
#         logger.info("Starting training...")

#     for epoch in range(args.num_epochs):
#         # 在分布式环境中，设置epoch给sampler
#         if args.distributed:
#             train_dataloader.sampler.set_epoch(epoch)
        
#         if not args.distributed or dist.get_rank() == 0:
#             logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

#         # 训练
#         train_metrics = train_epoch(
#             model, train_dataloader, optimizer, scheduler,
#             device, args.gradient_accumulation_steps, args.distributed
#         )

#         # 确保train_metrics中包含'total_loss'
#         if 'total_loss' not in train_metrics:
#             train_metrics['total_loss'] = 0.0

#         train_losses.append(train_metrics["total_loss"])

#         # 输出训练信息
#         if not args.distributed or dist.get_rank() == 0:
#             logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - "
#                         f"Training loss: {train_metrics['total_loss']:.4f}")

#         # 评估
#         eval_metrics = evaluate(model, test_dataloader, device)
#         eval_accuracies.append(eval_metrics["accuracy"])

#         if not args.distributed or dist.get_rank() == 0:
#             logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - "
#                         f"Evaluation accuracy: {eval_metrics['accuracy']:.4f}")
def main():
    try:
        # 解析参数
        args = ModelArgs()

        # 初始化分布式环境
        device = init_distributed_backend(args)

        # 设置随机种子
        set_seed(args.seed)

        # 只在主进程中创建输出目录和设置日志
        if not args.distributed or dist.get_rank() == 0:
            # 创建输出目录
            os.makedirs(args.output_dir, exist_ok=True)

            # 设置日志文件
            file_handler = logging.FileHandler(args.log_file)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

            logger.info("Starting Cultural Alignment Model Training")
            logger.info(f"Arguments: {args}")

        # 初始化模型
        if not args.distributed or dist.get_rank() == 0:
            logger.info("Initializing model...")
        
        model = CulturalAlignmentModel(args)
        model.to(device)

        # 检查模型是否有 llama_shared 属性
        if hasattr(model, 'llama_shared') and hasattr(model.llama_shared, 'tokenizer'):
            tokenizer = model.llama_shared.tokenizer
        else:
            # 如果没有 llama_shared 属性，尝试从其他地方获取 tokenizer
            # 检查模型是否有 tokenizer 属性
            if hasattr(model, 'tokenizer'):
                tokenizer = model.tokenizer
            else:
                # 如果都没有，尝试从模型的其他部分获取
                # 这取决于 CulturalAlignmentModel 的实现
                # 可能需要查看模型代码来确定如何获取 tokenizer
                raise AttributeError("Cannot find tokenizer in the model. Please check the model implementation.")
        
        # 在分布式环境中使用DDP包装模型
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )

        # 创建数据集
        if not args.distributed or dist.get_rank() == 0:
            logger.info("Loading datasets...")
        
        train_dataset = CulturalBenchDataset(
            args.train_data_path,
            tokenizer,
            args.max_length
        )

        test_dataset = CulturalBenchDataset(
            args.test_data_path,
            tokenizer,
            args.max_length
        )

        # 创建数据加载器
        train_dataloader = create_dataloader(
            train_dataset, 
            args.batch_size, 
            shuffle=True, 
            distributed=args.distributed
        )
        
        test_dataloader = create_dataloader(
            test_dataset, 
            args.batch_size, 
            shuffle=False, 
            distributed=args.distributed
        )

        # 设置优化器
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # 计算总步数
        total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

        # 设置学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )

        # 训练循环
        best_accuracy = 0
        train_losses = []
        eval_accuracies = []

        if not args.distributed or dist.get_rank() == 0:
            logger.info("Starting training...")

        for epoch in range(args.num_epochs):
            # 在分布式环境中，设置epoch给sampler
            if args.distributed:
                train_dataloader.sampler.set_epoch(epoch)
            
            if not args.distributed or dist.get_rank() == 0:
                logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

            # 训练
            train_metrics = train_epoch(
                model, train_dataloader, optimizer, scheduler,
                device, args.gradient_accumulation_steps, args.distributed
            )

            # 确保train_metrics中包含'total_loss'
            if 'total_loss' not in train_metrics:
                train_metrics['total_loss'] = 0.0

            train_losses.append(train_metrics["total_loss"])

            # 输出训练信息
            if not args.distributed or dist.get_rank() == 0:
                logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - "
                            f"Training loss: {train_metrics['total_loss']:.4f}")

            # 评估
            eval_metrics, predictions, labels = evaluate(model, test_dataloader, device, args.distributed)
            if eval_metrics:  # 只在主进程中评估
                eval_accuracies.append(eval_metrics.get("accuracy", 0))

                if not args.distributed or dist.get_rank() == 0:
                    logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - "
                                f"Evaluation accuracy: {eval_metrics.get('accuracy', 0):.4f}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 在分布式环境中，需要确保所有进程都退出
        if args.distributed:
            dist.destroy_process_group()
        
        raise e


if __name__ == "__main__":
    main()

# import json
# import logging
# import os
# import random
# from typing import Dict

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import torch
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# from torch.optim import AdamW
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from transformers import get_linear_schedule_with_warmup

# from model.args import ModelArgs
# from model.main import CulturalAlignmentModel

# # 设置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class CulturalBenchDataset(Dataset):
#     """文化对齐数据集"""

#     def __init__(self, data_path: str, tokenizer, max_length: int = 512):
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#         # 加载数据
#         with open(data_path, 'r', encoding='utf-8') as f:
#             self.data = json.load(f)

#         logger.info(f"Loaded {len(self.data)} samples from {data_path}")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         # 提取数据
#         prompt = item["prompt"]
#         query = item["query"]
#         response = item["response"]

#         # 构建输入文本
#         input_text = f"Question: {prompt}\nOption: {query}\nAnswer:"

#         # 标签转换：TRUE -> 1, FALSE -> 0
#         label = 1 if response == "TRUE" else 0

#         # Tokenize
#         encoding = self.tokenizer(
#             input_text,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_length,
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": encoding["input_ids"].squeeze(),
#             "attention_mask": encoding["attention_mask"].squeeze(),
#             "labels": torch.tensor(label, dtype=torch.long),
#             "prompt": prompt,
#             "query": query,
#             "response": response
#         }

# def set_seed(seed: int):
#     """设置随机种子"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
#     """创建数据加载器"""
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=4,
#         pin_memory=True
#     )

# def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
#     """计算评估指标"""
#     accuracy = accuracy_score(labels, predictions)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     }

# def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, save_path: str):
#     """绘制混淆矩阵"""
#     cm = confusion_matrix(labels, predictions)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['FALSE', 'TRUE'],
#                 yticklabels=['FALSE', 'TRUE'])
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# def train_epoch(model: CulturalAlignmentModel,
#                 dataloader: DataLoader,
#                 optimizer: torch.optim.Optimizer,
#                 scheduler: torch.optim.lr_scheduler._LRScheduler,
#                 device: str,
#                 gradient_accumulation_steps: int = 1) -> Dict[str, float]:
#     """训练一个epoch"""
#     model.train()
#     total_loss = 0
#     total_classification_loss = 0
#     total_load_balancing_loss = 0
#     total_diversity_loss = 0
#     num_batches = 0

#     progress_bar = tqdm(dataloader, desc="Training")

#     for step, batch in enumerate(progress_bar):
#         # 移动到设备
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         # 前向传播
#         outputs = model(input_ids, attention_mask, labels)

#         # 获取损失
#         loss = outputs["loss"]
#         classification_loss = outputs["classification_loss"]
#         load_balancing_loss = outputs["load_balancing_loss"]
#         diversity_loss = outputs["diversity_loss"]

#         # 梯度累积
#         loss = loss / gradient_accumulation_steps
#         loss.backward()

#         # 更新参数
#         if (step + 1) % gradient_accumulation_steps == 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()

#         # 累积损失
#         total_loss += loss.item() * gradient_accumulation_steps
#         total_classification_loss += classification_loss.item()
#         total_load_balancing_loss += load_balancing_loss.item()
#         total_diversity_loss += diversity_loss.item()
#         num_batches += 1

#         # 更新进度条
#         progress_bar.set_postfix({
#             'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
#             'lr': f'{scheduler.get_last_lr()[0]:.2e}'
#         })

#     return {
#         "total_loss": total_loss / num_batches,
#         "classification_loss": total_classification_loss / num_batches,
#         "load_balancing_loss": total_load_balancing_loss / num_batches,
#         "diversity_loss": total_diversity_loss / num_batches
#     }

# def evaluate(model: CulturalAlignmentModel,
#              dataloader: DataLoader,
#              device: str) -> Dict[str, float]:
#     """评估模型"""
#     model.eval()
#     all_predictions = []
#     all_labels = []
#     total_loss = 0
#     num_batches = 0

#     with torch.no_grad():
#         progress_bar = tqdm(dataloader, desc="Evaluating")

#         for batch in progress_bar:
#             # 移动到设备
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             # 前向传播
#             outputs = model(input_ids, attention_mask, labels)

#             # 获取预测
#             logits = outputs["logits"]
#             predictions = torch.argmax(logits, dim=-1)

#             # 收集结果
#             all_predictions.extend(predictions.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             total_loss += outputs["loss"].item()
#             num_batches += 1

#     # 计算指标
#     metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
#     metrics["loss"] = total_loss / num_batches

#     return metrics, all_predictions, all_labels

# def main():
#     # 解析参数
#     args = ModelArgs()

#     # 设置随机种子
#     set_seed(args.seed)

#     # 创建输出目录
#     os.makedirs(args.output_dir, exist_ok=True)

#     # 设置日志文件
#     file_handler = logging.FileHandler(args.log_file)
#     file_handler.setLevel(logging.INFO)
#     logger.addHandler(file_handler)

#     logger.info("Starting Cultural Alignment Model Training")
#     logger.info(f"Arguments: {args}")

#     # 初始化模型
#     logger.info("Initializing model...")
#     model = CulturalAlignmentModel(args)
#     model.to(args.device)

#     # 打印模型参数统计
#     param_stats = model.get_trainable_parameters()
#     logger.info(f"Trainable parameters: {param_stats}")

#     # 创建数据集
#     logger.info("Loading datasets...")
#     train_dataset = CulturalBenchDataset(
#         args.train_data_path,
#         model.llama_shared.tokenizer,
#         args.max_length
#     )

#     test_dataset = CulturalBenchDataset(
#         args.test_data_path,
#         model.llama_shared.tokenizer,
#         args.max_length
#     )

#     # 创建数据加载器
#     train_dataloader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
#     test_dataloader = create_dataloader(test_dataset, args.batch_size, shuffle=False)

#     # 设置优化器
#     optimizer = AdamW(
#         model.parameters(),
#         lr=args.learning_rate,
#         weight_decay=args.weight_decay
#     )

#     # 计算总步数
#     total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

#     # 设置学习率调度器
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=args.warmup_steps,
#         num_training_steps=total_steps
#     )

#     # 训练循环
#     best_accuracy = 0
#     train_losses = []
#     eval_accuracies = []

#     logger.info("Starting training...")

#     for epoch in range(args.num_epochs):
#         logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

#         # 训练
#         train_metrics = train_epoch(
#             model, train_dataloader, optimizer, scheduler,
#             args.device, args.gradient_accumulation_steps
#         )

#         logger.info(f"Train metrics: {train_metrics}")
#         train_losses.append(train_metrics["total_loss"])

#         # 评估
#         eval_metrics, predictions, labels = evaluate(model, test_dataloader, args.device)
#         logger.info(f"Eval metrics: {eval_metrics}")
#         eval_accuracies.append(eval_metrics["accuracy"])

#         # 保存最佳模型
#         if eval_metrics["accuracy"] > best_accuracy:
#             best_accuracy = eval_metrics["accuracy"]
#             model.save_model(args.model_save_path)
#             logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")

#             # 绘制混淆矩阵
#             plot_confusion_matrix(
#                 np.array(labels),
#                 np.array(predictions),
#                 os.path.join(args.output_dir, "confusion_matrix.png")
#             )

#     # 绘制训练曲线
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses)
#     plt.title('Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')

#     plt.subplot(1, 2, 2)
#     plt.plot(eval_accuracies)
#     plt.title('Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')

#     plt.tight_layout()
#     plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
#     plt.close()

#     logger.info(f"Training completed. Best accuracy: {best_accuracy:.4f}")
#     logger.info(f"Results saved to {args.output_dir}")

# if __name__ == "__main__":
#     main()

