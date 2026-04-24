from matplotlib import pyplot as plt


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
        plt.savefig(f'results/figures/{fig_name}.png', dpi=300)
    elif mode == 'show':
        plt.show()
    plt.close()
