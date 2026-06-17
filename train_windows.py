import os

mask_predformer = [False, True]

need_mask = [False, True]
need_mask = [need_mask[0]]

vars = ['ssh','uv']
vars = [vars[0]]

need_wind = [False,True]
need_wind = [need_wind[0]]

gated = [False, True]
gated = [gated[0]]

start_time = ['2020-01-01','1993-01-01']
start_time = [start_time[1]]

model_names = ['gsta','predformer','predrnn','tau']
model_names = model_names[3:]

pinn_lambdas= [0, 0.7]
pinn_lambdas = [pinn_lambdas[0]]

loss_ignore_nan = [False, True]
loss_ignore_nan = [loss_ignore_nan[1]]

shuffle = [False, True]
shuffle = [shuffle[1]]

num_runs = 0
for st in start_time:
    for model_name in model_names:
        print(f"{'*'*40} Model {model_name} {'*'*40}")
        for wind in need_wind:
            for var in vars:
                for mask in need_mask:
                    for g in gated:
                        if model_name == 'predformer' and g:
                            continue
                        for mask_pred in mask_predformer:
                            if model_name != 'predformer' and mask_pred:
                                continue
                            for p in pinn_lambdas:
                                for l in loss_ignore_nan:
                                    for s in shuffle:

                                        cmd_parts = [
                                                        "python -m tools.trainers",
                                                        "--batch_size_train 4",
                                                        f"--model_name {model_name}",
                                                        "--patch_size 8",
                                                        "--gradient_clip",
                                                        "--gradient_clip_value 1",

                                                        "--input_length 10",
                                                        "--output_length 10",
                                                        "--area indian",
                                                        "--env windows",
                                                        "--norm",
                                            "--SEED 1"
                                                    ]

                                        num_runs += 1
                                        cmd_parts.append(f"--start_time_train {st}")
                                        print(f"{'*'*40} Run {num_runs} {'*'*40}\n")
                                        if wind:
                                            cmd_parts.append("--need_wind")
                                        if var == 'ssh':
                                            cmd_parts.append("--need_ssh")
                                        if var == 'uv':
                                            cmd_parts.append("--need_uv")
                                        if mask:
                                            cmd_parts.append("--need_mask")
                                        if mask_pred:
                                            cmd_parts.append("--mask_predformer")

                                        if g:
                                            cmd_parts.append("--gated")

                                        if p != 0:
                                            cmd_parts.append("--is_pinn")
                                            cmd_parts.append(f"--pinn_lambda {p}")
                                        if l:
                                            cmd_parts.append("--loss_ignore_nan")
                                        if s:
                                            cmd_parts.append("--shuffle")

                                        cmd = " ".join(cmd_parts)
                                        os.system(cmd)
                                        print(f"over\n")
