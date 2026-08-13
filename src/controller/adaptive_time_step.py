import numpy as np


class AdaptiveTimeStepController:
    """
    面向齿轮-油膜-侧板耦合系统的自适应时间步长控制器，基于以下准则动态调整 dt
    1. 运动学预测防碰撞
    2. 多物理场指标（膜厚、挤压速度、结构加速度）
    """

    def __init__(
        self,
        dt_init: float = 1e-4,
        dt_min: float = 1e-5,
        dt_max: float = 1e-3,
        eta: float = 0.5,  # 防碰撞安全系数 (0~1)
        h_ref: float = 1e-5,  # 参考膜厚 [m]
        ht_ref: float = -1e-3,  # 参考挤压速度 [m/s]
        acc_ref: float = 1.0,  # 参考结构加速度 [m/s²]
        max_increase: float = 2.0,  # 单步最大放大因子
        max_decrease: float = 0.5,  # 单步最大缩小因子
        smoothing_alpha: float = 0.7,  # 历史平滑系数 (0~1)
    ):
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.eta = eta
        self.h_ref = h_ref
        self.ht_ref = ht_ref
        self.acc_ref = acc_ref
        self.max_inc = max_increase
        self.max_dec = max_decrease
        self.alpha = smoothing_alpha

        self.prev_dt = dt_init

    def compute_next_dt(
        self,
        h_cells: np.ndarray,
        ht_cells: np.ndarray,
        structural_vec: float,
        structural_acc: float,
    ):
        """
        根据当前步物理场计算下一步时间步长
        Args:
            h_cells: 各单元膜厚 (N,)
            ht_cells: 各单元挤压速度 ∂h/∂t (N,)
            structural_vec: 结构速度
            structural_acc: 结构加速度
        Returns:
            更新后的 dt
        """
        # 1. 提取关键指标
        h_min = np.min(h_cells)
        # ht_max = np.max(np.abs(ht_cells))

        # 2. 防碰撞步长阈值计算
        eps = 1e-12  # 防除零误差
        ds_max = self.eta * h_min
        v_abs = abs(structural_vec)
        a_abs = abs(structural_acc)
        if a_abs < eps:
            dt_collision = ds_max / (v_abs + eps)
        else:
            dt_collision = (
                -v_abs + np.sqrt(v_abs**2 + 2 * a_abs * ds_max)
            ) / (a_abs + eps)

        # # 3. 多准则缩放因子计算
        # # 膜厚准则：膜越薄，时间尺度越短 → dt ∝ h
        # f_h = h_min / max(self.h_ref, 1e-9)
        # f_h = np.clip(f_h, 0.05, 5.0)

        # # 挤压速度准则：∂h/∂t 越大，瞬态效应越强 → dt ∝ h/|∂h/∂t|
        # f_ht = self.h_ref / max(ht_max, 1e-9)
        # f_ht = np.clip(f_ht, 0.05, 5.0)

        # # 结构加速度准则：惯性力变化快时需加密步长 → dt ∝ 1/|a|
        # f_acc = 1.0
        # if structural_acc is not None:
        #     f_acc = self.acc_ref / max(abs(structural_acc), 1e-9)
        #     f_acc = np.clip(f_acc, 0.05, 5.0)

        # # 4. 保守策略：取最严格准则
        # raw_factor = min(f_h, f_ht, f_acc)

        # # 5. 融合两种准则，限制单步变化幅度（防震荡）
        # raw_dt = min(dt_collision, self.dt * raw_factor)
        raw_dt = dt_collision
        raw_dt = np.clip(raw_dt, self.dt * self.max_dec, self.dt * self.max_inc)
        raw_dt = np.clip(raw_dt, self.dt_min, self.dt_max)

        # 6. 指数平滑（工业 CFD 标准做法，避免 dt 剧烈跳变导致耦合发散）
        self.dt = self.alpha * self.prev_dt + (1.0 - self.alpha) * raw_dt
        self.prev_dt = self.dt

    def get_dt(self) -> float:
        return self.dt
