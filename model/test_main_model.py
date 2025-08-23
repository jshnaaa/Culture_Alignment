import torch
import torch.distributed as dist
import os
import tempfile
import numpy as np
from args import ModelArgs
from main import CulturalAlignmentModel

def init_distributed_backend(args):
    """初始化分布式环境"""
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)  # 将进程绑定到对应GPU
        args.distributed = True
    else:
        args.distributed = False
    
    # 设置设备
    if torch.cuda.is_available():
        if args.local_rank != -1:
            args.device = torch.device(f"cuda:{args.local_rank}")
        else:
            args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    
    return args.device

def create_test_data(args, batch_size=2, seq_len=64):
    """创建测试数据"""
    # 创建虚拟输入数据
    input_ids = torch.randint(1000, 2000, (batch_size, seq_len)).to(args.device)
    attention_mask = torch.ones((batch_size, seq_len)).to(args.device)
    labels = torch.randint(0, args.num_classes, (batch_size,)).to(args.device)
    
    return input_ids, attention_mask, labels

def test_forward_pass(model, args):
    """测试前向传播"""
    print("🧪 测试前向传播")
    
    # 创建测试数据
    input_ids, attention_mask, labels = create_test_data(args)
    
    try:
        # 前向传播
        outputs = model(input_ids, attention_mask, labels)
        
        # 检查输出是否包含所有必需的键
        required_keys = ["llama_features", "expert_weights", "router_logits", 
                        "expert_output", "logits", "classification_loss", 
                        "load_balancing_loss", "diversity_loss", "loss"]
        
        for key in required_keys:
            if key not in outputs:
                print(f"❌ 前向传播输出缺少键: {key}")
                return False
            else:
                print(f"✓ {key}: {outputs[key].shape if hasattr(outputs[key], 'shape') else outputs[key]}")
        
        # 检查损失是否为标量
        if outputs["loss"].dim() != 0:
            print("❌ 损失不是标量")
            return False
        
        print("✓ 前向传播测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predict_function(model, args):
    """测试预测函数"""
    print("\n🧪 测试预测函数")
    
    # 获取实际模型（如果使用了DDP）
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 创建测试数据
    input_ids, attention_mask, _ = create_test_data(args)
    
    try:
        # 预测
        predictions = actual_model.predict(input_ids, attention_mask)
        
        # 检查预测输出是否包含所有必需的键
        required_keys = ["predictions", "probabilities", "logits", "expert_weights"]
        
        for key in required_keys:
            if key not in predictions:
                print(f"❌ 预测输出缺少键: {key}")
                return False
            else:
                print(f"✓ {key}: {predictions[key].shape}")
        
        # 检查预测形状是否正确
        if predictions["predictions"].shape[0] != input_ids.shape[0]:
            print("❌ 预测数量与输入样本数量不匹配")
            return False
            
        if predictions["probabilities"].shape[1] != args.num_classes:
            print("❌ 概率数量与类别数量不匹配")
            return False
        
        print("✓ 预测函数测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 预测函数失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encode_and_predict(model, args):
    """测试端到端预测"""
    print("\n🧪 测试端到端预测")
    
    # 获取实际模型（如果使用了DDP）
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 创建测试文本
    test_texts = [
        "This is a test sentence for cultural alignment.",
        "Another example text to test the model's encoding and prediction capabilities."
    ]
    
    try:
        # 端到端预测
        results = actual_model.encode_and_predict(test_texts)
        
        # 检查结果
        if results["predictions"].shape[0] != len(test_texts):
            print("❌ 预测数量与输入文本数量不匹配")
            return False
            
        print(f"✓ 预测结果: {results['predictions'].cpu().numpy()}")
        print(f"✓ 概率分布形状: {results['probabilities'].shape}")
        print("✓ 端到端预测测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 端到端预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# def test_save_and_load(model, args):
#     """测试模型保存和加载"""
#     print("\n🧪 测试模型保存和加载")
    
#     # 只在主进程进行保存和加载测试
#     if args.distributed and dist.get_rank() != 0:
#         print("✓ 非主进程跳过保存加载测试")
#         return True
    
#     # 获取实际模型（如果使用了DDP）
#     actual_model = model.module if hasattr(model, 'module') else model
    
#     try:
#         # 创建临时目录
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # 保存模型
#             actual_model.save_model(temp_dir)
#             print("✓ 模型保存成功")
            
#             # 创建新模型实例
#             new_model = CulturalAlignmentModel(args)
#             new_model = new_model.to(args.device)
            
#             # 加载模型
#             new_model.load_model(temp_dir)
#             print("✓ 模型加载成功")
            
#             # 测试加载的模型是否能正常工作
#             input_ids, attention_mask, labels = create_test_data(args)
#             outputs = new_model(input_ids, attention_mask, labels)
            
#             if "loss" in outputs:
#                 print("✓ 加载的模型前向传播成功")
#                 return True
#             else:
#                 print("❌ 加载的模型前向传播失败")
#                 return False
                
#     except Exception as e:
#         print(f"❌ 保存/加载测试失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

def test_model_initialization():
    """测试模型初始化"""
    print("🚀 测试模型初始化")
    print("="*60)

    args = ModelArgs()
    args.llama_model_path = "../Meta-Llama-3.1-8B-Instruct"
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 添加调试信息
    print(f"LOCAL_RANK: {args.local_rank}")
    print(f"WORLD_SIZE: {args.world_size}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")

    # 确保分布式环境初始化
    args.device = init_distributed_backend(args)
    
    # 添加更多调试信息
    print(f"分布式模式: {args.distributed}")
    print(f"最终设备: {args.device}")

    try:
        print(f"创建文化对齐模型...")
        print(f"  Llama路径: {args.llama_model_path}")
        print(f"  设备: {args.device}")

        # 在分布式模式下，禁用量化和自动设备映射
        if args.distributed:
            args.use_quantization = False  # 禁用量化
            args.device_map = None  # 禁用自动设备映射
        
        model = CulturalAlignmentModel(args)

        # 将模型移动到正确的设备
        model = model.to(args.device)

        # 只在分布式环境下使用DDP
        if args.distributed:
            dist.barrier()  # 等待所有进程初始化完毕
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
            print(f"✓ 使用分布式数据并行 (DDP)")
        
        # 输出设备信息
        if not args.distributed or dist.get_rank() == 0:
            print(f"✓ 模型创建成功")
            print(f"当前设备：{args.device}")
            if torch.cuda.is_available():
                print(f"当前 GPU 内存占用: {torch.cuda.memory_allocated(args.device)} bytes")
                print(f"可用 GPU 数量: {torch.cuda.device_count()}")

        return model, args

    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_all_tests():
    """运行所有测试"""
    print(f"当前可用的 GPU 数量: {torch.cuda.device_count()}")
    model, args = test_model_initialization()

    if model is None:
        print(f"\n💔 模型初始化失败，无法继续测试")
        return False
    
    # 运行所有测试
    tests_passed = 0
    total_tests = 4
    
    # 测试前向传播
    if test_forward_pass(model, args):
        tests_passed += 1
    
    # 测试预测函数
    if test_predict_function(model, args):
        tests_passed += 1
    
    # 测试端到端预测
    if test_encode_and_predict(model, args):
        tests_passed += 1
    
    # 测试保存和加载
    if test_save_and_load(model, args):
        tests_passed += 1
    
    # 输出测试结果
    if not args.distributed or dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"测试结果: {tests_passed}/{total_tests} 通过")
        
        if tests_passed == total_tests:
            print("🎊 所有测试成功！")
            return True
        else:
            print("💔 部分测试失败")
            return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)