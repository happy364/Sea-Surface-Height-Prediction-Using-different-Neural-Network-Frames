import graphviz


def generate_loss_diagram():
    # 创建有向图
    dot = graphviz.Digraph(name='Geostrophic_Loss_Diagram', format='png')

    # 设置图的全局属性 (从上到下排布，字体)
    dot.attr(rankdir='TB', fontname='Helvetica', fontsize='12', splines='ortho')

    # 设置节点的全局属性 (圆角矩形，浅色背景)
    dot.attr('node', shape='box', style='filled, rounded', fillcolor='#f4f6f8', fontname='Helvetica', margin='0.3,0.15')

    # 1. 核心输入
    dot.node('Inputs', 'INPUTS:\nPredicted SSH (ζ_pred) & Target SSH (ζ_target)', fillcolor='#e2e8f0', shape='folder')

    # 2. 基础损失路径
    dot.node('BaseLoss', 'BASE SSH LOSS\nL_SSH = MSE(ζ_pred, ζ_target)', fillcolor='#dbeafe')

    # 3. 地转约束核心计算模块 (用虚线框包围起来)
    with dot.subgraph(name='cluster_geo') as c:
        c.attr(label='Latitude-Weighted Geostrophic Constraint', style='dashed', color='gray',
               fontname='Helvetica-Bold')

        # 3.1 空间梯度
        c.node('Gradient', '1. Spatial Gradient Calculation\n(Using Sobel Operators)')

        # 3.2 速度计算
        c.node('Velocity', '2. Geostrophic Velocity Calculation\nug = -(g/f)∂ζ/∂y,  vg = (g/f)∂ζ/∂x')

        # 3.3 归一化与重标度
        c.node('Normalization', '3. Normalization & Rescaling\nScale by (σ_SSH / σ_geo) to balance magnitudes')

        # 3.4 地转损失计算
        c.node('GeoLoss', '4. Geostrophic Loss Calculation\nL_geo = MSE_u + MSE_v')

        # 3.5 纬度加权
        c.node('LatWeight',
               '5. Latitude Weighting\nApply w(φ) to suppress low-latitude singularities\nw(φ) = √Sigmoid(k·(φ-φ0))',
               fillcolor='#fef08a')

        # 子图内的连接
        c.edge('Gradient', 'Velocity')
        c.edge('Velocity', 'Normalization')
        c.edge('Normalization', 'GeoLoss')
        c.edge('GeoLoss', 'LatWeight')

    # 4. 最终输出
    dot.node('TotalLoss', 'FINAL TOTAL LOSS\nL_total = L_SSH + λ · [w(φ) · L_geo]', fillcolor='#dcfce3',
             style='filled,bold', shape='ellipse')

    # 外部连接路径
    dot.edge('Inputs', 'BaseLoss', label=' Direct Path')
    dot.edge('Inputs', 'Gradient', label=' Constraint Path')

    dot.edge('BaseLoss', 'TotalLoss')
    dot.edge('LatWeight', 'TotalLoss', label=' Multiplied by λ')

    return dot


if __name__ == '__main__':
    # 生成图表对象
    diagram = generate_loss_diagram()

    # 保存并查看图片 (会在当前目录下生成 Geostrophic_Loss_Diagram.png)
    # view=True 表示生成后自动使用系统默认看图软件打开
    diagram.render(filename='Geostrophic_Loss_Diagram', cleanup=True, view=True)

    print("示意图已成功生成并保存为 Geostrophic_Loss_Diagram.png")