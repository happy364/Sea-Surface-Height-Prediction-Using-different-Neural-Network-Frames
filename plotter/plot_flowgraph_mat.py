import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_optimized_flowchart():
    # figsize: 控制画布大小（单位：英寸），直接影响整体比例
    # 👉 想让图更“松/大”：增大数值；更紧凑：减小
    fig, ax = plt.subplots(figsize=(14, 10))

    # 坐标轴范围：决定“布局坐标系”
    # 👉 所有元素的位置 (x,y) 都基于这个范围
    ax.set_xlim(0, 10)   # x轴范围（横向布局空间）
    ax.set_ylim(0, 10)   # y轴范围（纵向布局空间）

    ax.axis('off')  # 关闭坐标轴显示（做示意图必须关）

    # --- 统一样式配置 ---
    # boxstyle:
    #   'round'：圆角矩形
    #   pad：内边距（非常关键，控制“框内空白”）
    # 👉 pad越大 → 框越大（文字不变）
    BASE_BOX_STYLE = dict(
        boxstyle='round,pad=1.0',  # 👉 可以调到1.2~1.5让框更大更舒服
        linewidth=2.5              # 边框粗细
    )

    STEP_BOX_STYLE = dict(
        boxstyle='round,pad=0.7',  # 👉 步骤框稍小，建议 0.6~1.0
        facecolor='#FFFBEB',       # 背景色（可换）
        edgecolor='#D97706',       # 边框颜色
        linewidth=2                # 边框粗细
    )

    # arrowstyle:
    #   'simple'：实心箭头（推荐）
    # head_width / head_length：
    #   控制箭头头部大小（很关键）
    ARROW_STYLE = dict(
        arrowstyle='simple,head_width=1.2,head_length=1.2',
        color='#4A5568',
        lw=6  # 👉 箭头线宽（建议 1.5~3 之间调）
    )

    # 字体大小（统一控制视觉层级）
    TITLE_FONT = 24   # 👉 标题（可调到20更突出）
    MAIN_FONT = 22    # 👉 主模块
    STEP_FONT = 22    # 👉 子步骤（可调到14避免偏小）

    # =========================
    # 1. 输入层
    # =========================
    ax.text(
        5, 10.0,  # 👉 (x,y) 位置（控制布局）
        'SSH Predicted & Target\n$\\zeta_{p}, \\zeta_{t}$',
        ha='center',  # 水平居中
        va='center',  # 垂直居中
        fontsize=TITLE_FONT,
        fontweight='bold',
        bbox=dict(
            **BASE_BOX_STYLE,
            facecolor='#E2E8F0',  # 背景色
            edgecolor='#4A5568'   # 边框色
        )
    )

    # =========================
    # 2. 左侧 Loss
    # =========================
    ax.text(
        2.2, 5.0,  # 👉 左侧位置（调这个可以移动模块）
        'Base SSH Loss\n\n$L_{\\zeta} = MSE(\\zeta_{p}, \\zeta_{t})$',
        ha='center',
        va='center',
        fontsize=MAIN_FONT,
        bbox=dict(
            **BASE_BOX_STYLE,
            facecolor='#F0F4F8',
            edgecolor='#2D3748'
        )
    )

    # =========================
    # 3. 右侧约束模块（大框）
    # =========================
    rect = patches.FancyBboxPatch(
        (5.2, 2.0),   # 👉 左下角坐标
        4.3, 6.0,     # 👉 (宽, 高) —— 控制整体模块大小
        boxstyle="round,pad=0.2",  # 👉 外框 padding（一般不用太大）
        ec="#A0AEC0",  # 边框颜色
        fc="#F7FAFC",  # 填充颜色
        ls="--",       # 虚线
        lw=2,
        zorder=0       # 层级（0=最底层）
    )
    ax.add_patch(rect)

    # 模块标题
    ax.text(
        7.35, 7.6,
        'Geostrophic Constraint Module',
        ha='center',
        fontsize=MAIN_FONT,
        fontweight='bold',
        color='#2D3748'
    )

    # =========================
    # 内部步骤
    # =========================
    steps = [
        '1. Velocity Calculation\n$u_g, v_g \\propto \\frac{g}{f}\\nabla\\zeta$',
        '2. Latitude Weighting\n$L_{geo}^w = w(\\phi) \\cdot L_{geo}$',
        '3. Normalized Loss\n$L_{geo}^w = (\\sigma_{\\zeta}/\\sigma_{u,v})$'
    ]

    # 👉 控制每个步骤“纵向位置”（核心布局参数）
    # 👉 间距过大 → 稀疏；间距过小 → 拥挤
    step_y_pos = [6.7, 5.2, 3.8, 2.5]

    for i, text in enumerate(steps):
        ax.text(
            7.35, step_y_pos[i],
            text,
            ha='center',
            va='center',
            fontsize=STEP_FONT,
            fontweight='medium',
            bbox=STEP_BOX_STYLE
        )

        # 步骤之间箭头
        if i < len(steps) - 1:
            ax.annotate(
                '',
                xy=(7.35, step_y_pos[i + 1] + 0.5),  # 👉 箭头终点
                xytext=(7.35, step_y_pos[i] - 0.5),  # 👉 箭头起点
                arrowprops=dict(
                    arrowstyle='->',
                    color='#D97706',
                    lw=2,               # 👉 线宽（建议 ≥2）
                    mutation_scale=20   # 👉 箭头头大小（关键参数！）
                )
            )

    # =========================
    # 4. 总损失
    # =========================
    ax.text(
        5, 0.7,
        'FINAL TOTAL LOSS\n$L_{total} = L_{\\zeta} + \\lambda \\cdot L_{geo}^w$',
        ha='center',
        va='center',
        fontsize=TITLE_FONT,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=1.2',  # 👉 可以调到1.5增强视觉重量
            facecolor='#F0FFF4',
            edgecolor='#38A169',
            linewidth=3
        )
    )

    # =========================
    # 主连接箭头
    # =========================
    # 👉 这里控制“全局结构流向”

    # 输入 -> 左
    ax.annotate(
        '',
        xy=(2.2, 5.9),
        xytext=(4.3, 9.3),
        arrowprops=ARROW_STYLE
    )

    # 输入 -> 右
    ax.annotate(
        '',
        xy=(7.35, 8.2),
        xytext=(5.7, 9.3),
        arrowprops=ARROW_STYLE
    )

    # 左 -> 总损失
    ax.annotate(
        '',
        xy=(4.2, 1.4),
        xytext=(2.2, 4.1),
        arrowprops=ARROW_STYLE
    )

    # 右 -> 总损失
    ax.annotate(
        '',
        xy=(5.8, 1.4),
        xytext=(7.35, 1.7),
        arrowprops=ARROW_STYLE
    )

    # 自动调整边距（防止被裁剪）
    # 👉 如果你想更“紧凑”，可以尝试 pad=0.5
    plt.tight_layout()

    plt.show()


draw_optimized_flowchart()