import json
import logging
import os
import random
import traceback
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# 设置环境变量以减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def compute_js_divergence_metrics(pred_probs: np.ndarray, target_probs: np.ndarray) -> dict:
    """
    计算JS散度相关的评估指标

    Args:
        pred_probs: 预测的概率分布 [batch_size, num_classes]
        target_probs: 目标概率分布 [batch_size, num_classes]

    Returns:
        metrics: 包含JS散度指标的字典
    """
    # 确保概率分布归一化
    pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
    target_probs = target_probs / target_probs.sum(axis=1, keepdims=True)

    # 计算JS散度
    js_divergences = []
    for i in range(len(pred_probs)):
        p = pred_probs[i]
        q = target_probs[i]

        # 计算中间分布M = (P + Q) / 2
        m = (p + q) / 2

        # 计算KL散度，添加小的epsilon避免log(0)
        epsilon = 1e-8
        kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
        kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))

        # JS散度
        js_div = 0.5 * (kl_pm + kl_qm)
        js_divergences.append(js_div)

    js_divergences = np.array(js_divergences)

    # 计算1-JS距离作为相似度指标
    js_similarity = 1 - js_divergences

    return {
        "js_divergence": float(js_divergences.mean()),
        "js_similarity": float(js_similarity.mean()),
        "js_std": float(js_divergences.std()),
        "js_min": float(js_divergences.min()),
        "js_max": float(js_divergences.max())
    }

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

class CulturalDataset(Dataset):
    """通用文化对齐数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, dataset_type: str = "culturalbench"):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_type = dataset_type

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Loaded {len(self.data)} samples from {data_path} (dataset_type: {dataset_type})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.dataset_type == "culturalbench":
            return self._process_culturalbench_item(item)
        elif self.dataset_type == "globalopinions":
            return self._process_globalopinions_item(item)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _process_culturalbench_item(self, item):
        """处理CulturalBench数据集的样本"""
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

    def _process_globalopinions_item(self, item):
        """处理Global Opinions数据集的样本"""
        # 提取数据
        prompt = item["prompt"]
        query = item["query"]  # 选项列表的字符串表示
        response = item["response"]  # 概率分布字典

        # 构建输入文本
        # 解析选项列表
        try:
            options = eval(query) if isinstance(query, str) else query
            if isinstance(options, list):
                options_text = ", ".join(options)
            else:
                options_text = str(query)
        except:
            options_text = str(query)

        input_text = f"Question: {prompt}\nOptions: {options_text}\nAnswer:"

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 解析目标概率分布
        target_probs = self._parse_target_probabilities(response)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "target_probs": target_probs,
            "raw_response": response,
            "prompt": prompt,
            "query": query,
            "response": response
        }

    def _parse_target_probabilities(self, response):
        """解析目标概率分布"""
        try:
            # 使用json安全解析
            if isinstance(response, str):
                # 提取字典部分（移除defaultdict包装）
                dict_start = response.find('{')
                dict_end = response.rfind('}') + 1
                if dict_start >= 0 and dict_end > dict_start:
                    dict_str = response[dict_start:dict_end]
                    response_dict = json.loads(dict_str.replace("'", '"'))
                else:
                    raise ValueError("No dictionary found in response")
            else:
                response_dict = response

            # 合并所有国家的概率（取平均）
            country_probs = []
            for country, probs in response_dict.items():
                if isinstance(probs, list) and len(probs) == 5:  # Global Opinions有5个类别
                    # 验证概率值是否有效
                    probs_array = np.array(probs, dtype=np.float32)
                    if np.isnan(probs_array).any() or np.isinf(probs_array).any():
                        logger.warning(f"Invalid probabilities found for {country}: {probs}")
                        continue
                    if probs_array.sum() <= 0:
                        logger.warning(f"Zero or negative probability sum for {country}: {probs}")
                        continue
                    # 归一化概率
                    probs_array = probs_array / probs_array.sum()
                    country_probs.append(probs_array)

            if country_probs:
                # 计算所有国家的平均概率
                avg_probs = np.mean(country_probs, axis=0)
                # 确保概率分布有效
                if np.isnan(avg_probs).any() or np.isinf(avg_probs).any() or avg_probs.sum() <= 0:
                    logger.warning("Invalid average probabilities, using uniform distribution")
                    return torch.ones(5, dtype=torch.float32) / 5
                # 重新归一化
                avg_probs = avg_probs / avg_probs.sum()
                return torch.tensor(avg_probs, dtype=torch.float32)
            else:
                # 如果没有有效数据，返回均匀分布
                logger.warning("No valid country probabilities found, using uniform distribution")
                return torch.ones(5, dtype=torch.float32) / 5

        except Exception as e:
            logger.warning(f"Failed to parse target probabilities: {e}")
            return torch.ones(5, dtype=torch.float32) / 5

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

def print_predictions(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], step: int):
    """
    打印模型预测结果

    Args:
        outputs: 模型输出字典，包含logits等
        batch: 输入批次数据
        step: 当前训练步数
    """
    try:
        # 获取logits并计算预测概率
        logits = outputs["logits"]  # [batch_size, num_classes]
        predicted_probs = torch.softmax(logits, dim=-1)

        print(f"\n=== Step {step} - Predictions ===")

        if "target_probs" in batch:
            # GlobalOpinions数据集 - 概率分布预测
            target_probs = batch["target_probs"]
            batch_size = min(2, logits.size(0))  # 最多显示2个样本

            print("GlobalOpinions Dataset - Probability Distribution Prediction:")
            print("Classes: ['Very favorable', 'Somewhat favorable', 'Somewhat unfavorable', 'Very unfavorable', 'DK/Refused']")

            for i in range(batch_size):
                print(f"\nSample {i+1}:")
                if "prompt" in batch:
                    # 截断过长的prompt
                    prompt_text = batch["prompt"][i][:100] + "..." if len(batch["prompt"][i]) > 100 else batch["prompt"][i]
                    print(f"  Question: {prompt_text}")

                # 打印预测概率分布（保留3位小数）
                pred_values = predicted_probs[i].detach().cpu().numpy()
                target_values = target_probs[i].detach().cpu().numpy()

                print(f"  Predicted:  [{', '.join([f'{val:.3f}' for val in pred_values])}]")
                print(f"  Target:     [{', '.join([f'{val:.3f}' for val in target_values])}]")

                # 计算单个样本的JS散度
                pred_np = pred_values / pred_values.sum()  # 归一化
                target_np = target_values / target_values.sum()  # 归一化
                m = (pred_np + target_np) / 2
                epsilon = 1e-8

                kl_pm = np.sum(pred_np * np.log((pred_np + epsilon) / (m + epsilon)))
                kl_qm = np.sum(target_np * np.log((target_np + epsilon) / (m + epsilon)))
                js_div = 0.5 * (kl_pm + kl_qm)
                js_similarity = 1 - js_div

                print(f"  JS Divergence: {js_div:.6f}, JS Similarity: {js_similarity:.6f}")

                # 显示最高概率的预测类别
                pred_class = torch.argmax(predicted_probs[i]).item()
                target_class = torch.argmax(target_probs[i]).item()
                class_names = ['Very favorable', 'Somewhat favorable', 'Somewhat unfavorable', 'Very unfavorable', 'DK/Refused']
                print(f"  Predicted Class: {class_names[pred_class]} (prob: {pred_values[pred_class]:.3f})")
                print(f"  Target Class: {class_names[target_class]} (prob: {target_values[target_class]:.3f})")

                # 显示国家信息（如果有的话）
                if "country" in batch:
                    print(f"  Country: {batch['country'][i]}")

                # 显示原始响应（截断显示）
                if "raw_response" in batch:
                    raw_response = str(batch["raw_response"][i])[:150]
                    print(f"  Raw Response: {raw_response}...")

                # 显示概率分布的分类倾向
                favorable_prob = pred_values[0] + pred_values[1]  # Very + Somewhat favorable
                unfavorable_prob = pred_values[2] + pred_values[3]  # Very + Somewhat unfavorable
                neutral_prob = pred_values[4]  # DK/Refused

                target_favorable = target_values[0] + target_values[1]
                target_unfavorable = target_values[2] + target_values[3]
                target_neutral = target_values[4]

                print(f"  Predicted Sentiment: Favorable={favorable_prob:.3f}, Unfavorable={unfavorable_prob:.3f}, Neutral={neutral_prob:.3f}")
                print(f"  Target Sentiment:    Favorable={target_favorable:.3f}, Unfavorable={target_unfavorable:.3f}, Neutral={target_neutral:.3f}")

        elif "labels" in batch:
            # CulturalBench数据集 - 分类任务
            labels = batch["labels"]
            predictions = torch.argmax(logits, dim=-1)
            batch_size = min(2, logits.size(0))  # 最多显示2个样本

            print("CulturalBench Dataset - Binary Classification:")

            for i in range(batch_size):
                print(f"\nSample {i+1}:")
                if "prompt" in batch:
                    prompt_text = batch["prompt"][i][:100] + "..." if len(batch["prompt"][i]) > 100 else batch["prompt"][i]
                    print(f"  Question: {prompt_text}")
                if "query" in batch:
                    print(f"  Option: {batch['query'][i]}")

                # 打印预测结果
                pred_prob = predicted_probs[i].detach().cpu().numpy()
                pred_label = predictions[i].item()
                true_label = labels[i].item()

                print(f"  Predicted: {'TRUE' if pred_label == 1 else 'FALSE'} (confidence: {pred_prob[pred_label]:.3f})")
                print(f"  Actual:    {'TRUE' if true_label == 1 else 'FALSE'}")
                print(f"  Correct:   {'✓' if pred_label == true_label else '✗'}")

        print("=" * 50)

    except Exception as e:
        logger.warning(f"Error printing predictions at step {step}: {e}")
        # 不抛出异常，避免影响训练

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
    nan_detected = False

    # 只在主进程中显示进度条
    if not distributed or dist.get_rank() == 0:
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader

    for step, batch in enumerate(progress_bar):
        # 移动到设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 根据数据集类型处理不同的输入
        if "labels" in batch:
            # CulturalBench数据集
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask, labels=labels)
        elif "target_probs" in batch:
            # Global Opinions数据集
            target_probs = batch["target_probs"].to(device)
            outputs = model(input_ids, attention_mask, target_probs=target_probs)
        else:
            raise ValueError("Unknown batch format")

        # 检查输出是否有NaN
        if torch.isnan(outputs["loss"]).any():
            nan_detected = True
            if not distributed or dist.get_rank() == 0:
                logger.warning(f"NaN detected in loss at step {step}, skipping batch")
            # 跳过当前批次，但仍然递增num_batches以避免除零错误
            num_batches += 1
            continue

        # 获取损失
        loss = outputs["loss"]
        classification_loss = outputs["classification_loss"]
        load_balancing_loss = outputs["load_balancing_loss"]
        diversity_loss = outputs["diversity_loss"]

        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # 检查梯度是否有NaN或Inf
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    if not distributed or dist.get_rank() == 0:
                        logger.warning(f"NaN/Inf detected in gradients for {name} at step {step}")
                    break
        
        if has_nan_grad:
            if not distributed or dist.get_rank() == 0:
                logger.warning(f"Skipping parameter update due to NaN/Inf gradients at step {step}")
            optimizer.zero_grad()
            continue

        # 更新参数
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪（使用更小的阈值）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 累积损失
        total_loss += loss.item() * gradient_accumulation_steps
        total_classification_loss += classification_loss.item()
        total_load_balancing_loss += load_balancing_loss.item()
        total_diversity_loss += diversity_loss.item()
        num_batches += 1

        # 打印预测结果（每5步打印一次，观察训练过程）
        if step % 5 == 0 and (not distributed or dist.get_rank() == 0):
            print_predictions(outputs, batch, step)

        # 更新进度条
        if not distributed or dist.get_rank() == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

    # 如果检测到NaN，记录警告
    if nan_detected and (not distributed or dist.get_rank() == 0):
        logger.warning("NaN values detected during training. Consider reducing learning rate or using gradient clipping.")


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


def evaluate(model: CulturalAlignmentModel,
             dataloader: DataLoader,
             device: str,
             distributed: bool = False) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_target_probs = []
    total_loss = 0
    num_batches = 0

    # 获取数据集配置
    # dataset_config = model.dataset_config
    if distributed:
        dataset_config = model.module.dataset_config  # 通过.module访问原始模型
    else:
        dataset_config = model.dataset_config

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

            # 根据数据集类型处理不同的输入
            if "labels" in batch:
                # CulturalBench数据集
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask, labels=labels)

                # 获取预测和概率
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)

                # 收集结果
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                # 为CulturalBench创建空的目标概率占位符
                dummy_target_probs = torch.zeros_like(probabilities)
                all_target_probs.extend(dummy_target_probs.cpu().numpy())

            elif "target_probs" in batch:
                # Global Opinions数据集 - 概率分布预测任务
                target_probs = batch["target_probs"].to(device)
                outputs = model(input_ids, attention_mask, target_probs=target_probs)

                # 获取预测概率分布
                logits = outputs["logits"]
                probabilities = torch.softmax(logits, dim=-1)

                # 收集概率分布数据
                all_probabilities.extend(probabilities.cpu().numpy())
                all_target_probs.extend(target_probs.cpu().numpy())

                # 为了兼容现有结构，使用argmax作为参考预测（但主要评估是概率分布相似性）
                predictions = torch.argmax(probabilities, dim=-1)
                pseudo_labels = torch.argmax(target_probs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(pseudo_labels.cpu().numpy())

            else:
                raise ValueError("Unknown batch format")

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

        # 收集损失和批次数量
        losses = torch.tensor([total_loss, num_batches], dtype=torch.float32, device=device)
        dist.reduce(losses, dst=0, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            # 合并所有进程的结果
            all_predictions = torch.cat(gathered_predictions).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()
            total_loss, total_num_batches = losses.tolist()

            # 根据数据集类型计算指标
            if dataset_config["dataset_type"] == "globalopinions":
                # 对于Global Opinions，需要收集概率分布数据
                # 注意：这里需要额外的代码来收集概率分布数据
                # 由于代码复杂性，这里暂时使用分类指标作为参考
                metrics = compute_metrics(all_predictions, all_labels)
                metrics["note"] = "Using classification metrics as reference for GlobalOpinions in distributed mode"
            else:
                # 对于CulturalBench，使用传统分类指标
                metrics = compute_metrics(all_predictions, all_labels)

            metrics["loss"] = total_loss / total_num_batches if total_num_batches > 0 else 0

            return metrics, all_predictions, all_labels
        else:
            return {}, [], []
    else:
        # 非分布式环境：根据数据集类型计算指标
        if dataset_config["dataset_type"] == "globalopinions":
            # 对于Global Opinions数据集，计算JS散度指标
            probabilities_array = np.array(all_probabilities)
            target_probs_array = np.array(all_target_probs)
            metrics = compute_js_divergence_metrics(probabilities_array, target_probs_array)

            # 也计算分类指标作为参考
            classification_metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
            metrics.update({f"ref_{k}": v for k, v in classification_metrics.items()})
        else:
            # 对于CulturalBench数据集，计算传统分类指标
            metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))

        metrics["loss"] = total_loss / num_batches
        
        return metrics, all_predictions, all_labels

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
                find_unused_parameters=False  # 设置为False以减少内存使用
            )

        # 创建数据集
        if not args.distributed or dist.get_rank() == 0:
            logger.info("Loading datasets...")
            logger.info(f"Dataset type: {args.dataset_type}")
        
        train_dataset = CulturalDataset(
            args.train_data_path,
            tokenizer,
            args.max_length,
            args.dataset_type
        )

        test_dataset = CulturalDataset(
            args.test_data_path,
            tokenizer,
            args.max_length,
            args.dataset_type
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
            weight_decay=args.weight_decay,
            eps=1e-8,  # 添加小的epsilon值防止除零
            betas=(0.9, 0.999),  # 使用标准的beta值
            amsgrad=False  # 不使用AMSGrad变体
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

