import os

from bokeh.io import output_file


def merge_codes_to_single_md(target_dir, output_file='project_code_bundle.md'):
    # 配置支持的后缀及高亮标签
    ext_map = {'.py': 'python', '.sh': 'bash'}

    # 统计信息
    count = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(f"# 项目代码概览\n")
        outfile.write(f"生成时间: 2026-03-19\n")
        outfile.write(f"根目录路径: `{os.path.abspath(target_dir)}`\n\n---\n\n")

        for root, dirs, files in os.walk(target_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ext_map:
                    file_path = os.path.join(root, file)
                    # 计算相对路径，方便阅读
                    rel_path = os.path.relpath(file_path, target_dir)
                    lang = ext_map[ext]

                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            content = infile.read()

                        # 写入 Markdown 结构
                        outfile.write(f"## 文件: {rel_path}\n")
                        outfile.write(f"路径: `{file_path}`\n\n")
                        outfile.write(f"```{lang}\n")
                        outfile.write(content)
                        outfile.write("\n```\n\n---\n\n")

                        print(f"已加入合并列表: {rel_path}")
                        count += 1
                    except Exception as e:
                        print(f"读取 {rel_path} 失败: {e}")

    print(f"\n✅ 成功！已将 {count} 个文件合并至: {output_file}")


# 使用方法：将 'your_project_path' 换成你的代码文件夹路径
projetct_path = r'D:\MVPfore\analyze_combined'
output_file = projetct_path + r'\project_code_bundle.md'
merge_codes_to_single_md(projetct_path, output_file )