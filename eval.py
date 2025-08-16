import argparse
import json
import logging
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.args import ModelArgs
from model.main import CulturalAlignmentModel
from train import CulturalBenchDataset, create_dataloader, compute_metrics, plot_confusion_matrix

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model: CulturalAlignmentModel, args: ModelArgs):
        self.model = model
        self.args = args
        self.device = args.device

    def evaluate_on_dataset(self, dataloader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """在数据集上评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_expert_weights = []
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating")

            for batch in progress_bar:
                # 移动到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播
                outputs = self.model(input_ids, attention_mask, labels)

                # 获取预测结果
                logits = outputs["logits"]
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                expert_weights = outputs["expert_weights"]

                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_expert_weights.extend(expert_weights.cpu().numpy())

                total_loss += outputs["loss"].item()
                num_batches += 1

        # 计算指标
        predictions_array = np.array(all_predictions)
        labels_array = np.array(all_labels)
        probabilities_array = np.array(all_probabilities)
        expert_weights_array = np.array(all_expert_weights)

        metrics = compute_metrics(predictions_array, labels_array)
        metrics["loss"] = total_loss / num_batches

        return metrics, predictions_array, labels_array, probabilities_array, expert_weights_array

    def analyze_expert_behavior(self, expert_weights: np.ndarray, labels: np.ndarray, save_dir: str):
        """分析专家行为模式"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. 专家利用率分析
        mean_utilization = expert_weights.mean(axis=0)
        std_utilization = expert_weights.std(axis=0)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        expert_ids = range(len(mean_utilization))
        plt.bar(expert_ids, mean_utilization, yerr=std_utilization, capsize=5)
        plt.title('Expert Utilization (Mean ± Std)')
        plt.xlabel('Expert ID')
        plt.ylabel('Utilization Rate')
        plt.xticks(expert_ids)

        for i, (mean, std) in enumerate(zip(mean_utilization, std_utilization)):
            plt.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', fontsize=9)

        # 2. 专家权重分布
        plt.subplot(1, 2, 2)
        plt.boxplot([expert_weights[:, i] for i in expert_ids], labels=[f'E{i}' for i in expert_ids])
        plt.title('Expert Weight Distribution')
        plt.xlabel('Expert ID')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'expert_analysis.png'))
        plt.close()

        # 3. 按标签分析专家使用
        true_indices = labels == 1
        false_indices = labels == 0

        true_weights = expert_weights[true_indices].mean(axis=0)
        false_weights = expert_weights[false_indices].mean(axis=0)

        plt.figure(figsize=(10, 6))
        x = np.arange(len(expert_ids))
        width = 0.35

        plt.bar(x - width/2, true_weights, width, label='TRUE', alpha=0.8)
        plt.bar(x + width/2, false_weights, width, label='FALSE', alpha=0.8)

        plt.title('Expert Utilization by Label')
        plt.xlabel('Expert ID')
        plt.ylabel('Average Utilization')
        plt.xticks(x, [f'Expert {i}' for i in expert_ids])
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'expert_by_label.png'))
        plt.close()

        # 保存数值结果
        analysis_results = {
            'mean_utilization': mean_utilization.tolist(),
            'std_utilization': std_utilization.tolist(),
            'true_label_weights': true_weights.tolist(),
            'false_label_weights': false_weights.tolist()
        }

        with open(os.path.join(save_dir, 'expert_analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)

        return analysis_results

    def analyze_prediction_confidence(self, probabilities: np.ndarray, labels: np.ndarray,
                                    predictions: np.ndarray, save_dir: str):
        """分析预测置信度"""
        os.makedirs(save_dir, exist_ok=True)

        # 计算置信度（最大概率）
        confidence = np.max(probabilities, axis=1)
        correct_predictions = (predictions == labels)

        plt.figure(figsize=(15, 5))

        # 1. 置信度分布
        plt.subplot(1, 3, 1)
        plt.hist(confidence[correct_predictions], bins=30, alpha=0.7, label='Correct', density=True)
        plt.hist(confidence[~correct_predictions], bins=30, alpha=0.7, label='Incorrect', density=True)
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.legend()

        # 2. 准确率 vs 置信度
        plt.subplot(1, 3, 2)
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []

        for i in range(len(confidence_bins) - 1):
            mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i + 1])
            if mask.sum() > 0:
                bin_accuracy = correct_predictions[mask].mean()
                bin_count = mask.sum()
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.plot(bin_centers, bin_accuracies, 'o-')
        plt.title('Accuracy vs Confidence')
        plt.xlabel('Confidence Bin')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)

        # 3. 类别概率分布
        plt.subplot(1, 3, 3)
        true_probs = probabilities[labels == 1, 1]  # TRUE类的概率
        false_probs = probabilities[labels == 0, 0]  # FALSE类的概率

        plt.hist(true_probs, bins=30, alpha=0.7, label='TRUE samples', density=True)
        plt.hist(false_probs, bins=30, alpha=0.7, label='FALSE samples', density=True)
        plt.title('Class Probability Distribution')
        plt.xlabel('Probability of Predicted Class')
        plt.ylabel('Density')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'))
        plt.close()

        # 保存置信度统计
        confidence_stats = {
            'mean_confidence': float(confidence.mean()),
            'std_confidence': float(confidence.std()),
            'mean_confidence_correct': float(confidence[correct_predictions].mean()),
            'mean_confidence_incorrect': float(confidence[~correct_predictions].mean()),
            'accuracy_by_confidence': list(zip(bin_centers.tolist(), bin_accuracies, bin_counts))
        }

        with open(os.path.join(save_dir, 'confidence_analysis.json'), 'w') as f:
            json.dump(confidence_stats, f, indent=2)

        return confidence_stats

    def generate_detailed_report(self, metrics: Dict[str, float],
                               predictions: np.ndarray, labels: np.ndarray,
                               save_path: str):
        """生成详细的评估报告"""

        # 分类报告
        class_names = ['FALSE', 'TRUE']
        report = classification_report(
            labels, predictions,
            target_names=class_names,
            output_dict=True
        )

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)

        # 创建报告
        report_content = f"""
# Cultural Alignment Model Evaluation Report

## Overall Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1']:.4f}
- **Loss**: {metrics['loss']:.4f}

## Per-Class Performance

### FALSE Class
- Precision: {report['FALSE']['precision']:.4f}
- Recall: {report['FALSE']['recall']:.4f}
- F1-Score: {report['FALSE']['f1-score']:.4f}
- Support: {report['FALSE']['support']}

### TRUE Class
- Precision: {report['TRUE']['precision']:.4f}
- Recall: {report['TRUE']['recall']:.4f}
- F1-Score: {report['TRUE']['f1-score']:.4f}
- Support: {report['TRUE']['support']}

## Confusion Matrix
```
                Predicted
                FALSE  TRUE
Actual FALSE     {cm[0,0]:4d}  {cm[0,1]:4d}
       TRUE      {cm[1,0]:4d}  {cm[1,1]:4d}
```

## Summary
- Total samples: {len(labels)}
- Correct predictions: {(predictions == labels).sum()}
- Incorrect predictions: {(predictions != labels).sum()}
- Error rate: {1 - metrics['accuracy']:.4f}
"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return report_content

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cultural Alignment Model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved model")
    parser.add_argument("--test_data", type=str,
                       default="data/CulturalBench-Hard_test.json",
                       help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    log_file = os.path.join(args.output_dir, "evaluation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info("Starting model evaluation...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output directory: {args.output_dir}")

    # 加载模型参数
    model_args = ModelArgs()
    model_args.batch_size = args.batch_size

    # 初始化和加载模型
    logger.info("Loading model...")
    model = CulturalAlignmentModel(model_args)
    model.load_model(args.model_path)
    model.to(model_args.device)

    # 加载测试数据
    logger.info("Loading test dataset...")
    test_dataset = CulturalBenchDataset(
        args.test_data,
        model.llama_shared.tokenizer,
        model_args.max_length
    )

    test_dataloader = create_dataloader(
        test_dataset,
        args.batch_size,
        shuffle=False
    )

    # 初始化评估器
    evaluator = ModelEvaluator(model, model_args)

    # 执行评估
    logger.info("Running evaluation...")
    metrics, predictions, labels, probabilities, expert_weights = evaluator.evaluate_on_dataset(test_dataloader)

    logger.info(f"Evaluation metrics: {metrics}")

    # 生成混淆矩阵
    plot_confusion_matrix(
        labels, predictions,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )

    # 专家行为分析
    logger.info("Analyzing expert behavior...")
    expert_analysis = evaluator.analyze_expert_behavior(
        expert_weights, labels,
        os.path.join(args.output_dir, "expert_analysis")
    )

    # 置信度分析
    logger.info("Analyzing prediction confidence...")
    confidence_analysis = evaluator.analyze_prediction_confidence(
        probabilities, labels, predictions,
        os.path.join(args.output_dir, "confidence_analysis")
    )

    # 生成详细报告
    logger.info("Generating evaluation report...")
    report = evaluator.generate_detailed_report(
        metrics, predictions, labels,
        os.path.join(args.output_dir, "evaluation_report.md")
    )

    # 保存所有结果到JSON
    results = {
        "metrics": metrics,
        "expert_analysis": expert_analysis,
        "confidence_analysis": confidence_analysis,
        "model_parameters": model.get_trainable_parameters()
    }

    with open(os.path.join(args.output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # 打印总结
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*50)

    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()

