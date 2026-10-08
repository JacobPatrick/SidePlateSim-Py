# SidePlateSim

## 一键配置 conda 环境

```bash
conda env create -f environment.yml
```

## 侧板瞬态运动

## 端面边界泄漏

```python
pressure = solver.solve(film_param)
# 泊肃叶压差流量，每条边的积分值，单位 m³/s。
q_pressure = solver.calc_leakage(
    pressure.p, film_param, include_couette=False
)
# 默认还包含 Couette 拖曳流。
q_total = solver.calc_leakage(pressure.p, film_param)
# 比较沿边界的分布时，使用单位边长流量，单位 m²/s。
q_density = solver.calc_leakage(
    pressure.p, film_param, per_unit_length=True
)
```

返回列表与 `mesh.facets` 一一对应，正值表示流入油膜，负值表示流出。
同一齿腔的总流量可按 `mesh.facet_markers` 选取后求和。内部 facet
返回零；marker 为 0 或超出 `bc_lst` 范围的外边界沿用默认边界压力
`bc_lst[0]`，与压力组装一致。膜厚非正的所属单元不产生边界泄漏。

压差流量复用 Reynolds 方程的边界传导系数，无需求全场梯度。
Couette 项在边界积分点重构膜厚；`FilmParameterCalculator` 自动提供
`U_grad`，使刚体旋转速度也能重构到边界。手工构造 `FilmParam` 时，
可选的 `U_grad[a, b]` 表示 `dU_a/dx_b`；未提供时使用所属单元速度。

`plot_leak_rate(mesh, q_total, ...)` 接收 **m³/s 的积分流量**，默认除以
边长后绘制，避免网格边长不均产生显示跳变。不要将 `q_density` 再传入
这个默认绘图接口。设置 `per_unit_length=False` 可显示原始积分流量。

当前压力模型仍使用原有两点通量离散及负压截断。本次修复使边界压差
通量与该离散方程一致，但不替代非正交修正或质量守恒空化模型；负压
截断后的压力不能保证全域质量守恒。几何拐角和不同压力段的交界仍可能
产生实际的大流量梯度，不对这些结果进行人为平滑。
