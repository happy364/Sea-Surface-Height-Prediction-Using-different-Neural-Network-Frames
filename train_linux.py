import os

mask_predformer = [False, True]
mask_predformer = [mask_predformer[0]]

need_mask = [False, True]
need_mask = [need_mask[1]]

vars = ['ssh','uv']
vars = [vars[0]]


need_wind = [False,True]
need_wind = [need_wind[0]]

norms = [False, True]
norms = [norms[1]]

gated = [False, True]
gated = [gated[0]]

start_time = ['2021-01-01','1993-01-01']
start_time = [start_time[1]]

model_names = [ 'predrnn', 'predformer','gsta','rest', 'sted','tau']
# model_names = ['gsta','predrnn','predformer']

# model_names = ['predformer','predrnn']
model_names = model_names[0:1]

pinn_lambdas= [0, 0.1, 5.0, 0.5, 1.0, 0.7]
pinn_lambdas = [0.7, 0, 0.1, 5]

loss_ignore_nan = [False, True]
loss_ignore_nan = [loss_ignore_nan[1]]

shuffle = [False, True]
shuffle = [shuffle[1]]

patches = [8,4,2]
patches = [2]

num_runs = 0
for st in start_time:
    for norm in norms:
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
                                            for patch in patches:

                                                cmd_parts = [
                                                            "python3 -m tools.trainers",
                                                            "--batch_size_train 8",
                                                            f"--model_name {model_name}",
                                                            f"--patch_size {patch}",
                                                            "--gradient_clip",
                                                            "--gradient_clip_value 1",
                                                            "--env linux",
                                                            # "--area indian",
                                                            "--input_length 10",
                                                            "--output_length 10",
                                                            "--end_time_test 2026-06-14"
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

                                                if norm:
                                                    cmd_parts.append("--norm")

                                                cmd = " ".join(cmd_parts)
                                                os.system(cmd)
                                                print(f"over\n")
