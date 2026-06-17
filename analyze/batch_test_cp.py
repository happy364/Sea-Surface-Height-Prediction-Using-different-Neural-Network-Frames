
from thop import profile  # 需要安装: pip install thop
import torch
import csv
import os
from configs import parse_args, get_my_config
# from test_cp import benchmark_model
from models import PredFormer_Model, SimVP_Model, RNN

def benchmark_model(model, input_size, device="cuda"):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)

    # 1. 计算理论复杂度
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 2. 预热
    for _ in range(100):  # 批量测试时可适当减少预热次数提高效率
        _ = model(dummy_input)

    # 3. 测试延迟
    repetitions = 1000
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

    avg_latency = sum(timings) / repetitions
    throughput = (input_size[0] * 1000) / avg_latency

    # 4. 显存占用
    memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # 返回结果字典
    return {
        "Params(M)": round(params / 1e6, 4),
        "FLOPs(G)": round(flops / 1e9, 4),
        "Latency(ms)": round(avg_latency, 2),
        "Throughput(fps)": round(throughput, 2),
        "Memory(MB)": round(memory_reserved, 2)
    }



# 定义参数配置图
CONFIG_MAP = {
    'gsta': {
        'ultra_small': {"hid_S": 6, "hid_T": 64,  "N_T": 2, "mlp_ratio": 4.0, },
        'small': {"hid_S": 16, "hid_T": 128,  "N_T": 4, "mlp_ratio": 8.0, },
        'large': {"hid_S": 32, "hid_T": 256,  "N_T": 10, "mlp_ratio": 8.0, }
    },
    'predrnn': {
        'ultra_small': {'num_layers': 2, 'hidden_dim': 32, 'layer_norm': False},
        'small': {'num_layers': 3, 'hidden_dim': 64, 'layer_norm': True},
        'large': {'num_layers': 6, 'hidden_dim': 128, 'layer_norm': True}
    },
    'predformer': {
        'ultra_small': {'dim': 64, 'heads': 4, 'scale_dim': 4, 'depth': 1, 'Ndepth': 2},
        'small': {'dim': 128, 'heads': 4, 'scale_dim': 4, 'depth': 2, 'Ndepth': 4},
        'large': {'dim': 256, 'heads': 8, 'scale_dim': 8, 'depth': 2, 'Ndepth': 4}
    }
}


def main():
    args_base = parse_args()  # 获取基础配置
    args_base.height, args_base.width = 160, 160
    args_base.need_ssh = True  # 单通道海表高度
    args_base.need_mask = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_file = "/data/hjj/SEJ/model_paras_aviso_0.125deg_final/b1_mask_model_performance_results.csv"
    headers = ["Model", "Scale", "Params(M)", "FLOPs(G)", "Latency(ms)", "Throughput(fps)", "Memory(MB)"]

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for m_name in ['gsta', 'predrnn', 'predformer']:
            args_base.model_name = m_name
            for scale in ['ultra_small', 'small', 'large']:
                print(f"正在测试: {m_name} - {scale}...")

                m_config = CONFIG_MAP[m_name][scale]

                args = get_my_config(args_base, model_config=m_config)

                # 实例化
                if m_name == 'gsta':
                    raw_model = SimVP_Model(**args.model_config)
                elif m_name == 'predrnn':
                    raw_model = RNN(args.model_config)
                elif m_name == 'predformer':
                    raw_model = PredFormer_Model(args.model_config)


                # 运行测试
                input_size = (1, 10, 2, 160,160) if m_name != 'predrnn' else (1, 20, 128, 20, 20)
                metrics = benchmark_model(raw_model, input_size, device=device)

                # 保存数据
                metrics.update({"Model": m_name, "Scale": scale})
                writer.writerow(metrics)
                f.flush()  # 实时写入防止程序崩溃丢失数据


                torch.cuda.empty_cache()


    print(f"所有测试完成，结果已保存至: {csv_file}")


if __name__ == "__main__":
    main()

