import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri as mtri
from matplotlib import cm, colors
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
            ax.plot(xi, yi, 'r-', linewidth=1.5)

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


def plot_pressure_distribution(mesh, p_cells, fig_name, contour='True', mode='save'):
    points = np.array(mesh.points)
    elements = np.array(mesh.elements)
    centroids = np.mean(points[elements], axis=1)

    _, ax = plt.subplots(figsize=(6, 6))
    x_lst = [points[i][0] for i in range(len(points))]
    y_lst = [points[i][1] for i in range(len(points))]
    triang = mtri.Triangulation(x_lst, y_lst, triangles=elements)
    c = ax.tripcolor(triang, facecolors=p_cells, cmap='jet', shading='flat')
    c.set_clim(vmin=p_cells.min(), vmax=p_cells.max())  # 设置 colorbar 范围
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label='Pressure [Pa]')

    if contour == 'True':
        cx = centroids[:, 0]
        cy = centroids[:, 1]

        # 将单元中心点坐标与压力值配对，绘制等高线
        ax.tricontour(cx, cy, p_cells, levels=10, colors='black', linewidths=0.5, alpha=0.5)

    if mode == 'save':
        plt.savefig(
            f'results/figures/{fig_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            dpi=300,
        )
    elif mode == 'show':
        plt.show()
    plt.close()


def plot_leak_rate(mesh, leak_rate, fig_name, mode='save'):
    _, ax = plt.subplots(figsize=(7, 6))
    facets = np.array(mesh.facets)

    vmin = float(np.min(leak_rate))
    vmax = float(np.max(leak_rate))
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max  # 设置对称的 colorbar 范围
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('jet')

    for idx, facet_points in enumerate(facets):
        x = [mesh.points[j][0] for j in facet_points]
        y = [mesh.points[j][1] for j in facet_points]
        color = cmap(norm(leak_rate[idx]))
        ax.plot(x, y, color=color, linewidth=1)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Leak Rate [m^3/s]')
    if mode == 'save':
        plt.savefig(
            f'results/figures/{fig_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            dpi=300,
        )
    elif mode == 'show':
        plt.show()
    plt.close()
