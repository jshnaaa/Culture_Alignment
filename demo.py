#!/usr/bin/env python3
"""
Cultural Alignment Model Demo Script
演示如何使用训练好的模型进行文化对齐预测
"""

import argparse
import json
from typing import List, Dict

import numpy as np

from model.args import ModelArgs
from model.main import CulturalAlignmentModel


class CulturalAlignmentPredictor:
    """文化对齐预测器"""

    def __init__(self, model_path: str):
        """
        初始化预测器

        Args:
            model_path: 模型保存路径
        """
        self.args = ModelArgs()

        # 加载模型
        print("Loading Cultural Alignment Model...")
        self.model = CulturalAlignmentModel(self.args)
        self.model.load_model(model_path)
        self.model.to(self.args.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.args.device}")

        # 打印模型参数统计
        param_stats = self.model.get_trainable_parameters()
        print(f"Model parameters: {param_stats['total_trainable']:,} trainable / {param_stats['llama_total']:,} total")

    def predict_single(self, prompt: str, query: str) -> Dict:
        """
        对单个样本进行预测

        Args:
            prompt: 文化问题描述
            query: 选项或说明

        Returns:
            预测结果字典
        """
        # 使用模型进行预测
        result = self.model.encode_and_predict([prompt], [query])

        # 处理结果
        prediction = result["predictions"][0].item()
        probabilities = result["probabilities"][0].cpu().numpy()
        expert_weights = result["expert_weights"][0].cpu().numpy()

        # 转换为可读格式
        prediction_label = "TRUE" if prediction == 1 else "FALSE"
        confidence = float(np.max(probabilities))

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "probabilities": {
                "FALSE": float(probabilities[0]),
                "TRUE": float(probabilities[1])
            },
            "expert_weights": expert_weights.tolist(),
            "dominant_expert": int(np.argmax(expert_weights))
        }

    def predict_batch(self, samples: List[Dict[str, str]]) -> List[Dict]:
        """
        批量预测

        Args:
            samples: 包含prompt和query的样本列表

        Returns:
            预测结果列表
        """
        prompts = [sample["prompt"] for sample in samples]
        queries = [sample["query"] for sample in samples]

        # 批量预测
        result = self.model.encode_and_predict(prompts, queries)

        predictions = result["predictions"].cpu().numpy()
        probabilities = result["probabilities"].cpu().numpy()
        expert_weights = result["expert_weights"].cpu().numpy()

        # 处理结果
        results = []
        for i in range(len(samples)):
            prediction_label = "TRUE" if predictions[i] == 1 else "FALSE"
            confidence = float(np.max(probabilities[i]))

            results.append({
                "prompt": prompts[i],
                "query": queries[i],
                "prediction": prediction_label,
                "confidence": confidence,
                "probabilities": {
                    "FALSE": float(probabilities[i][0]),
                    "TRUE": float(probabilities[i][1])
                },
                "expert_weights": expert_weights[i].tolist(),
                "dominant_expert": int(np.argmax(expert_weights[i]))
            })

        return results

def interactive_demo(predictor: CulturalAlignmentPredictor):
    """交互式演示"""
    print("\n" + "="*60)
    print("CULTURAL ALIGNMENT MODEL - INTERACTIVE DEMO")
    print("="*60)
    print("Enter cultural questions and options to test the model.")
    print("Type 'quit' to exit, 'examples' to see sample inputs.\n")

    examples = [
        {
            "prompt": "In Japanese culture, what is traditionally said before starting to eat at a dinner table? The country corresponding to this question is Japan.",
            "query": "\"Itadakimasu\"",
            "expected": "TRUE"
        },
        {
            "prompt": "In Chinese culture, what should be avoided during Ghost Festival in Summertime in July? The country corresponding to this question is Hong Kong.",
            "query": "Go swimming",
            "expected": "TRUE"
        },
        {
            "prompt": "In German culture, what is the proper way to say \"cheers\" when you're drinking with someone? The country corresponding to this question is Germany.",
            "query": "Salud!",
            "expected": "FALSE"
        }
    ]

    while True:
        try:
            user_input = input("Enter command (predict/examples/quit): ").strip().lower()

            if user_input == 'quit':
                print("Thanks for using the Cultural Alignment Model!")
                break

            elif user_input == 'examples':
                print("\n--- EXAMPLE INPUTS ---")
                for i, example in enumerate(examples, 1):
                    result = predictor.predict_single(example["prompt"], example["query"])

                    print(f"\nExample {i}:")
                    print(f"Prompt: {example['prompt']}")
                    print(f"Query: {example['query']}")
                    print(f"Expected: {example['expected']}")
                    print(f"Predicted: {result['prediction']} (confidence: {result['confidence']:.3f})")
                    print(f"Dominant Expert: {result['dominant_expert']}")

                    # 显示专家权重
                    expert_weights = result['expert_weights']
                    print("Expert weights:", end=" ")
                    for j, weight in enumerate(expert_weights):
                        print(f"E{j}: {weight:.3f}", end=" ")
                    print()

            elif user_input == 'predict':
                # 获取用户输入
                print("\n--- MAKE PREDICTION ---")
                prompt = input("Enter the cultural question (prompt): ").strip()
                if not prompt:
                    print("Prompt cannot be empty!")
                    continue

                query = input("Enter the option/statement (query): ").strip()
                if not query:
                    print("Query cannot be empty!")
                    continue

                # 进行预测
                print("\nPredicting...")
                result = predictor.predict_single(prompt, query)

                # 显示结果
                print("\n--- PREDICTION RESULT ---")
                print(f"Prompt: {prompt}")
                print(f"Query: {query}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Probabilities - FALSE: {result['probabilities']['FALSE']:.3f}, TRUE: {result['probabilities']['TRUE']:.3f}")
                print(f"Dominant Expert: {result['dominant_expert']}")

                # 显示专家权重分布
                expert_weights = result['expert_weights']
                print("Expert Weight Distribution:")
                for i, weight in enumerate(expert_weights):
                    bar = "█" * int(weight * 20)  # 简单的文本条形图
                    print(f"  Expert {i}: {weight:.3f} {bar}")
                print()

            else:
                print("Invalid command. Use 'predict', 'examples', or 'quit'.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def batch_demo(predictor: CulturalAlignmentPredictor, input_file: str):
    """批量预测演示"""
    print(f"Loading test samples from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # 取前10个样本进行演示
    demo_samples = samples[:10] if len(samples) > 10 else samples

    print(f"Running batch prediction on {len(demo_samples)} samples...")

    # 准备输入
    batch_input = [
        {"prompt": sample["prompt"], "query": sample["query"]}
        for sample in demo_samples
    ]

    # 批量预测
    results = predictor.predict_batch(batch_input)

    # 显示结果
    print("\n" + "="*80)
    print("BATCH PREDICTION RESULTS")
    print("="*80)

    correct = 0
    for i, (sample, result) in enumerate(zip(demo_samples, results)):
        actual = sample.get("response", "UNKNOWN")
        predicted = result["prediction"]
        is_correct = actual == predicted
        if is_correct:
            correct += 1

        print(f"\nSample {i+1}:")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"Query: {sample['query']}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted} (confidence: {result['confidence']:.3f})")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print(f"Dominant Expert: {result['dominant_expert']}")

    accuracy = correct / len(demo_samples)
    print(f"\nBatch Accuracy: {accuracy:.3f} ({correct}/{len(demo_samples)})")

def main():
    parser = argparse.ArgumentParser(description="Cultural Alignment Model Demo")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"],
                       default="interactive",
                       help="Demo mode: interactive or batch")
    parser.add_argument("--input_file", type=str,
                       default="data/CulturalBench-Hard_test.json",
                       help="Input file for batch mode")

    args = parser.parse_args()

    try:
        # 初始化预测器
        predictor = CulturalAlignmentPredictor(args.model_path)

        if args.mode == "interactive":
            interactive_demo(predictor)
        elif args.mode == "batch":
            batch_demo(predictor, args.input_file)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model path is correct and the model was trained properly.")

if __name__ == "__main__":
    main()

