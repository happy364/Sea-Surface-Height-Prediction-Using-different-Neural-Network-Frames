import torch
import time
from thop import profile  # 需要安装: pip install thop


def benchmark_model(model, input_size, device="cuda"):
    model.to(device)
    model.eval()

    # 构造模拟输入
    dummy_input = torch.randn(input_size).to(device)

    # 1. 计算理论复杂度 (FLOPs & Params)
    # 使用 thop 工具库，比手动计算更准确
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"理论计算量: {flops / 1e9:.4f} GFLOPs")
    print(f"总参数量: {params / 1e6:.4f} M")

    # 2. 预热 (Warm-up)
    # GPU 首次启动需要加载内核，预热能排除干扰项
    print("正在预热...")
    for _ in range(200):
        _ = model.predict(dummy_input)

    # 3. 测试延迟 (Latency) & 吞吐量 (Throughput)
    print("开始性能测试...")
    repetitions = 1000
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()

            # 等待 GPU 完成所有任务
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 单位毫秒
            timings.append(curr_time)

    avg_latency = sum(timings) / repetitions
    batch_size = input_size[0]
    throughput = (batch_size * 1000) / avg_latency

    print(f"平均延迟: {avg_latency:.2f} ms")
    print(f"吞吐量: {throughput:.2f} samples/s")

    # 4. 显存占用 (Memory)
    memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    print(f"最大显存保留: {memory_reserved:.2f} MB")

if __name__ == "__main__":
    from tools.base_method import Model
    from models import PredFormer_Model, Mask_PredFormer_Model, SimVP_Model, RNN, ReST
    from configs import parse_args,get_my_config
    import numpy as np

    args_ = parse_args()

    args = get_my_config(args_)

    mask_land = torch.from_numpy(np.load(args.path_land_mask))  # (H,W) 1: invalid, 0: valid

    if args.model_name == 'predformer':
        if args.mask_predformer:
            model = Mask_PredFormer_Model(args.model_config, mask_land)
        else:
            model = PredFormer_Model(args.model_config)
    elif args.model_name == 'gsta' or args.model_name == 'tau':
        model = SimVP_Model(**args.model_config)
    elif args.model_name == 'predrnn':
        model = RNN(args.model_config)
    elif args.model_name == 'rest':
        model = ReST(**args.model_config)


    # model.load_state_dict(torch.load(model_para_path, weights_only=True))

    Model = Model(model, args)
    benchmark_model(Model, (1,10 if args.model_name != 'predrnn' else 20, 2 if args.need_mask else 1,160,160 ,))
