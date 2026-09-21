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

    def compute_next_dt(
        self,
        h_cells: np.ndarray,
        ht_cells: np.ndarray,
        structural_vec: float,
        structural_acc: float,
        accepted_dt: float | None = None,
    ) -> float:
        """
        根据当前步物理场计算下一步时间步长
        Args:
            h_cells: 各单元膜厚 (N,)
            ht_cells: 各单元挤压速度 ∂h/∂t (N,)
            structural_vec: 结构速度
            structural_acc: 结构加速度
            accepted_dt: 刚刚接受的物理时间步
        Returns:
            更新后的 dt
        """
        base_dt = self.dt if accepted_dt is None else accepted_dt

        # 1. 只对膜厚正在减小的单元施加防碰撞限制。ht > 0 表示
        # 两表面远离，不应像旧实现那样因取绝对值而缩短时间步。
        h_cells = np.asarray(h_cells)
        ht_cells = np.asarray(ht_cells)
        if h_cells.shape != ht_cells.shape:
            raise ValueError("h_cells 和 ht_cells 的形状必须一致")
        eps = 1e-12
        closing_speed = np.maximum(-ht_cells, 0.0)
        valid = (h_cells > 0.0) & (closing_speed > eps)

        # 结构加速度只有指向闭合方向时才构成附加限制。
        closing_acc = max(-structural_acc, 0.0)
        if np.any(valid):
            allowed_displacement = self.eta * h_cells[valid]
            cell_speed = closing_speed[valid]
            if closing_acc > eps:
                dt_cells = (
                    -cell_speed
                    + np.sqrt(
                        cell_speed**2 + 2.0 * closing_acc * allowed_displacement
                    )
                ) / closing_acc
            else:
                dt_cells = allowed_displacement / cell_speed
            dt_collision = float(np.min(dt_cells))
        elif np.any(h_cells <= 0.0):
            dt_collision = self.dt_min
        else:
            dt_collision = np.inf

        raw_dt = min(dt_collision, base_dt * self.max_inc)
        raw_dt = np.clip(raw_dt, self.dt_min, self.dt_max)

        # 2. 增大时间步时使用平滑；安全限制要求减小时立即生效。
        smoothed_dt = self.alpha * base_dt + (1.0 - self.alpha) * raw_dt
        if raw_dt < base_dt:
            smoothed_dt = raw_dt
        self.dt = float(np.clip(smoothed_dt, self.dt_min, self.dt_max))
        return self.dt

    def reject_step(self, rejected_dt: float) -> float:
        """记录一次失败尝试，并返回用于同一物理时刻重试的步长。"""
        self.dt = max(self.dt_min, rejected_dt * self.max_dec)
        return self.dt

    def get_dt(self) -> float:
        return self.dt
