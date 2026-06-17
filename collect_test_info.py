import os
import re
import csv
from pathlib import Path


# ==============================
# 正则解析规则（只需要在这里加变量）
# ==============================

PATTERNS = {
    "model_type": r"model_type:\s*(\S+)",
    "model": r"Model:\s*(\S+)",
    "inputs": r"Inputs:\s*([^\n]+)",
    "loss_function": r"Loss Function:\s*([^\n]+)",
    "test_loss": r"Test loss:\s*([\d.]+)",
    "best_val_loss": r"Best validation loss:\s*([\d.]+)",
    "patch_size": r"patch_size:\s*(\d+)",
    "batch_size": r"Batch Size:\s*(\d+)",
    "gradient_clipping": r"Gradient Clipping:\s*([^\n]+)",
    "optimizer": r"Optimizer:\s*(\S+)",
    "total_parameters": r"Total Parameters:\s*([\d,]+)",
    "norm": r"norm:\s*(\S+)",
    "pinn": r"pinn:\s*(\S+)",
}


# ==============================
# 单文件解析
# ==============================

def parse_log_file(log_file):

    info = {k: None for k in PATTERNS}

    info.update({
        "file_name": Path(log_file).name,
        "folder": Path(log_file).parent.name,
        "final_model_type": None,
        "epochs_trained": None,
        "total_time": None
    })

    try:

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # ---------- 正则批量解析 ----------
        for key, pattern in PATTERNS.items():

            match = re.search(pattern, content)

            if not match:
                continue

            value = match.group(1).strip()

            if key in ["test_loss", "best_val_loss"]:
                value = float(value)

            if key == "batch_size":
                value = int(value)

            if key == "patch_size":
                value = int(value)

            if key == "total_parameters":
                value = int(value.replace(",", ""))

            if key == "inputs":
                value = re.sub(r"\s+", "", value)

            info[key] = value

        # ---------- 确定模型类型 ----------
        info["final_model_type"] = info["model_type"] or info["model"]

        # ---------- 训练轮数 ----------
        early_stop = re.search(
            r"Early stopping triggered at epoch (\d+)", content)

        if early_stop:
            info["epochs_trained"] = int(early_stop.group(1))

        else:
            epochs = re.findall(
                r"Epoch (\d+)/\d+", content)

            if epochs:
                info["epochs_trained"] = int(epochs[-1])

        # ---------- 总时间 ----------
        time_match = re.search(
            r"Total time:\s*([\dhms ]+)", content)

        if time_match:
            info["total_time"] = time_match.group(1).strip()

    except Exception as e:
        print(f"解析失败 {log_file}: {e}")

    return info


# ==============================
# 收集所有日志
# ==============================

def collect_logs(folder, output_csv, std=1.):

    log_files = []

    for root, _, files in os.walk(folder):
        for f in files:
            if f.startswith("training") and f.endswith(".log"):
                log_files.append(os.path.join(root, f))

    print(f"找到 {len(log_files)} 个日志")

    results = [parse_log_file(f) for f in log_files]

    results.sort(key=lambda x: (
        (x["final_model_type"] or "").lower(),
        x["file_name"]
    ))

    # ==============================
    # CSV字段
    # ==============================

    fields = [
        "模型类型",
        "文件名",
        "所在文件夹",
        "输入",
        "patch_size",
        "损失函数",
        "批次大小",
        "梯度裁剪",
        "Norm",
        "PINN",
        "总参数量(M)",
        "优化器",
        "测试损失",
        "最佳验证损失",
        "训练轮数",
        "总训练时间",
        "备注"
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()

        current_model = None

        for info in results:

            new_model = (
                current_model is not None
                and current_model != info["final_model_type"]
            )

            current_model = info["final_model_type"]

            if new_model:
                writer.writerow({})

            params = (
                f"{info['total_parameters']/1e6:.2f}M"
                if info["total_parameters"]
                else "N/A"
            )

            row = {

                "模型类型": info["final_model_type"] or "N/A",
                "文件名": info["file_name"],
                "所在文件夹": info["folder"],
                "输入": info["inputs"] or "N/A",
                "损失函数": info["loss_function"] or "N/A",
                "patch_size": info["patch_size"] or "N/A",
                "批次大小": info["batch_size"] or "N/A",
                "梯度裁剪": info["gradient_clipping"] or "N/A",
                "Norm": info["norm"] or "N/A",
                "PINN": info["pinn"] or "N/A",
                "总参数量(M)": params,
                "优化器": info["optimizer"] or "N/A",
                "测试损失": f"{info['test_loss'] * std *100:.2f}" if info["test_loss"] else "N/A",
                "最佳验证损失": f"{info['best_val_loss'] *100 * std:.2f}" if info["best_val_loss"] else "N/A",
                "训练轮数": info["epochs_trained"] or "N/A",
                "总训练时间": info["total_time"] or "N/A",
                "备注": "↑ 新模型类型 ↑" if new_model else ""

            }

            writer.writerow(row)

    print("CSV 已生成:", output_csv)

    return output_csv

if __name__ == "__main__":

   folder_path = r"/data/hjj/ssh_prediction/work_dir/scs/RNN_seed42/patchsize"
   csv_path = os.path.join(folder_path, "model_performance_results.csv")
   collect_logs(folder_path, csv_path,std = 0.119)