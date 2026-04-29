import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri as mtri
from datetime import datetime


def plot_shapely_poly(poly, fig_name, mode='save'):
    """绘制 Shapely 多边形对象"""
    if poly.is_empty:
        print("几何为空，请检查输入数据")
        return

    # 兼容 Polygon 与 MultiPolygon
    polygons = poly.geoms if poly.geom_type == 'MultiPolygon' else [poly]

    _, ax = plt.subplots(figsize=(6, 6))
    for p in polygons:
        x, y = p.exterior.xy
        ax.plot(x, y, 'b-', linewidth=1.5)

        for interior in p.interiors:
            xi, yi = interior.xy
            ax.plot(xi, yi, 'b-', linewidth=1.5)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()

    if mode == 'save':
        plt.savefig(
            f'results/figures/{fig_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            dpi=300,
        )
    elif mode == 'show':
        plt.show()
    plt.close()


def plot_mesh(mesh, fig_name, mode='save'):
    """可视化 meshpy 生成的三角网格"""
    points = np.array(mesh.points)
    elements = np.array(mesh.elements)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.triplot(points[:, 0], points[:, 1], elements, 'b-', lw=0.5)
    ax.plot(points[:, 0], points[:, 1], 'ro', markersize=3, alpha=0.5)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    plt.tight_layout()
    if mode == 'save':
        plt.savefig(
            f'results/figures/{fig_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            dpi=300,
        )
    elif mode == 'show':
        plt.show()
    plt.close()


def plot_pressure_distribution(mesh, p_cells, fig_name, mode='save'):
    _, ax = plt.subplots(figsize=(6, 6))
    x_lst = [mesh.points[i][0] for i in range(len(mesh.points))]
    y_lst = [mesh.points[i][1] for i in range(len(mesh.points))]
    triang = mtri.Triangulation(x_lst, y_lst, triangles=mesh.elements)
    c = ax.tripcolor(triang, facecolors=p_cells, cmap='jet', shading='flat')
    c.set_clim(vmin=p_cells.min(), vmax=p_cells.max())  # 设置 colorbar 范围
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label='Pressure [Pa]')
    if mode == 'save':
        plt.savefig(
            f'results/figures/{fig_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            dpi=300,
        )
    elif mode == 'show':
        plt.show()
    plt.close()
