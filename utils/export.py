from meshpy.triangle import MeshInfo, write_gnuplot_mesh


def export_mesh(mesh: MeshInfo, filename: str, dir: str = 'results/mesh/'):
    """将网格几何信息导出为 .dat 文本文件"""
    try:
        write_gnuplot_mesh(dir + filename + '.dat', mesh)
        print(f'网格信息成功导出至 {filename}.dat')
    except Exception as e:
        print(f'导出网格信息时出错: {e}')
