import numpy as np
from interface.types import SidePlateState, FilmParam
from utils.math_tools import quaternion_to_euler

DRIVE_GEAR_CENTER = (0.0305, 0)
SLAVE_GEAR_CENTER = (-0.0305, 0)


def calc_film_params(mesh, state: SidePlateState, p_lst, omega, gear_type):
    """
    根据给定的网格、侧板状态和齿轮参数，计算油膜参数表
    """
    points = np.array(mesh.points)
    elements = np.array(mesh.elements)
    centroids = np.mean(points[elements], axis=1)

    roll, pitch, _ = quaternion_to_euler(*state.q)

    # 1. 计算节点处的油膜厚度
    h_cells = (
        -np.sin(pitch) * np.array([point[0] for point in centroids])
        + np.sin(roll) * np.array([point[1] for point in centroids])
        + state.p[2] * np.ones(len(centroids))
    )
    # 非负检查
    # assert np.any(
    #     h_cells > 0
    # ), "警告: 油膜厚度存在非正值，请检查齿轮位姿参数设置！"
    if np.any(h_cells <= 0):
        print("警告: 油膜厚度存在非正值！")
        # h_cells = np.clip(h_cells, 0.0, None)  # 截断油膜厚度负值
    # 油膜厚度梯度 (∂h/∂x, ∂h/∂y)
    h_grad = (-np.sin(pitch), np.sin(roll))

    # 2. 确定边界条件
    bc_lst = [p_lst[0]]
    for _, p_val in p_lst[1:]:
        bc_lst.append(p_val)

    # 3. 计算三角网格中心处的相对运动速度
    if gear_type == "drive":
        U_cells = np.array(
            [
                [
                    -omega * point[1],
                    omega * (point[0] - DRIVE_GEAR_CENTER[0]),
                ]
                for point in centroids
            ]
        )
    else:
        U_cells = np.array(
            [
                [
                    omega * point[1],
                    -omega * (point[0] - SLAVE_GEAR_CENTER[0]),
                ]
                for point in centroids
            ]
        )

    # 4. 计算三角网格中心处的挤压速度（两表面相互远离为正）
    ht_cells = (
        state.v[2] * np.ones(len(centroids))
        + np.cos(roll)
        * state.w[0]
        * np.array([point[1] for point in centroids])
        - np.cos(pitch)
        * state.w[1]
        * np.array([point[0] for point in centroids])
    )

    return FilmParam(
        h_cells=h_cells,
        h_grad=h_grad,
        U_cells=U_cells,
        ht_cells=ht_cells,
        bc_lst=bc_lst,
    )
