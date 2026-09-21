import atexit
from pathlib import Path


class SimulationLogger:
    """低频、缓冲写入的仿真状态日志。"""

    def __init__(self, path, every_steps=100, flush_every_records=10):
        if every_steps < 1:
            raise ValueError("every_steps 必须大于等于 1")
        self.every_steps = every_steps
        self.flush_every_records = flush_every_records
        self.records_since_flush = 0
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = log_path.open("a", encoding="utf-8")
        atexit.register(self.close)

    def write_step(
        self,
        step_index,
        time,
        dt,
        retry_count,
        solve_info,
        state,
    ):
        if step_index != 1 and step_index % self.every_steps != 0:
            return False

        self.file.write(f"时间: {time * 1000:.6f}ms\n")
        self.file.write(
            f"时间步: {dt * 1e6:.6f}us, 重试次数: {retry_count}\n"
        )
        self.file.write(
            f"迭代次数: {solve_info['num_iter']}, "
            f"残差: {solve_info['res_norm']:.3e}\n"
        )
        self.file.write(
            f"侧板受力: 主动轮 F={solve_info['F_drive']:.2f}N, "
            f"从动轮 F={solve_info['F_slave']:.2f}N\n"
        )
        self.file.write(f"侧板受合力矩: M={solve_info['M']}N·m\n")
        self.file.write(f"侧板受力: F={solve_info['F_side_plate']:.2f}N\n")
        self.file.write(
            f"侧板状态: p={state.p}, v={state.v}, "
            f"q={state.q}, w={state.w}\n\n"
        )
        self.records_since_flush += 1
        if self.records_since_flush >= self.flush_every_records:
            self.file.flush()
            self.records_since_flush = 0
        return True

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()
