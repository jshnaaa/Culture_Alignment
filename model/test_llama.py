import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch


class MyModel:
    def __init__(self, model_path):
        # 如果传入的是字符串，直接使用它作为模型路径
        if isinstance(model_path, str):
            self.model_path = model_path
        else:
            # 如果传入的是对象，尝试获取 llama_model_path 属性
            self.model_path = getattr(model_path, 'llama_model_path', None)
            if self.model_path is None:
                raise ValueError("Provided object does not have 'llama_model_path' attribute")

        # 加载本地Llama3.1模型（内存优化版）
        logging.info(f"Loading Llama model from {self.model_path}")

        # 自动加载与 Llama3.1 模型配套的分词器（tokenizer）
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # 加载模型配置
        config = AutoConfig.from_pretrained(self.model_path)

        # 优先尝试以8位量化（节省内存）方式加载Llama3.1模型，若失败则自动回退到float32精度加载
        try:
            # 首先尝试8位量化（如果可用）
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=torch.float16,  # 使用float16可以进一步节省显存
                device_map="auto",  # 自动将模型分配到多个GPU
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True,  # 开启8位量化
                offload_folder="./offload"  # 如果显存不足，将部分权重卸载到磁盘
            )
            logging.info("Successfully loaded with 8-bit quantization")
        except Exception as e:
            logging.warning(f"8-bit quantization failed: {e}, trying alternative loading...")
            # 如果8位量化失败，使用float32精度加载
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=torch.float32,  # 使用float32可能更稳定
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logging.info("Successfully loaded with float32 precision")


# 现在可以直接传入字符串路径
model = MyModel("../Meta-Llama-3.1-8B-Instruct")