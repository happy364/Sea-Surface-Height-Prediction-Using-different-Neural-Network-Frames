import os


def update_log_files(root_dir):
    # 目标字符串
    target_phrase = "Training Configuration:"
    # 要添加的内容
    append_text = " pinn=0.0"

    # 遍历母文件夹及其所有子文件夹
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # 只处理以 .log 结尾的文件
            if file.endswith(".log"):
                file_path = os.path.join(subdir, file)

                try:
                    # 读取原始文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # 检查是否需要修改并执行修改
                    modified = False
                    new_lines = []
                    for line in lines:
                        # 如果这一行包含目标短语，且尚未被修改（防止重复运行脚本导致多次添加）
                        if target_phrase in line and append_text not in line:
                            # 去掉行尾换行符，加上新内容，再补回换行符
                            new_line = line.rstrip('\n\r') + append_text + '\n'
                            new_lines.append(new_line)
                            modified = True
                        else:
                            new_lines.append(line)

                    # 如果文件发生了变化，写回文件
                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.writelines(new_lines)
                        print(f"已成功更新文件: {file_path}")
                    else:
                        print(f"跳过（无需修改或已修改）: {file_path}")

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

# 使用示例：将 'your_parent_folder_path' 替换为您实际的母文件夹路径
update_log_files('/data/hjj/SEJ/model_paras_aviso_0.125deg_final/log_summary')