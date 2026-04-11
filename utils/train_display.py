import sys
import time
from typing import Optional


class NullLiveDisplay:
    def start(self):
        return None

    def update(self, *args, **kwargs):
        return None

    def stop(self):
        return None


class RichTrainingDisplay:
    def __init__(self, total_steps: int, title: str = "Diff-Img2Img Training"):
        from rich.console import Console
        from rich.live import Live

        self.total_steps = total_steps
        self.title = title
        self.console = Console(stderr=False)
        self.live = Live(console=self.console, refresh_per_second=4, transient=False)
        self.start_time = time.perf_counter()

    def start(self):
        self.live.start()

    def _fmt(self, value, digits=4):
        if value is None:
            return "-"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    def _format_duration(self, seconds: Optional[float]) -> str:
        if seconds is None:
            return "-"
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def update(self, state: dict):
        from rich.columns import Columns
        from rich.panel import Panel
        from rich.table import Table

        step = state.get("step", 0)
        elapsed = time.perf_counter() - self.start_time
        eta = None
        if step > 0 and self.total_steps > step:
            eta = (elapsed / step) * (self.total_steps - step)

        summary = Table.grid(expand=True)
        summary.add_column(justify="left")
        summary.add_column(justify="left")
        summary.add_row("Phase", self._fmt(state.get("phase")))
        summary.add_row("Epoch", self._fmt(state.get("epoch")))
        summary.add_row("Step", f"{step}/{self.total_steps}")
        summary.add_row("Elapsed", self._format_duration(elapsed))
        summary.add_row("ETA", self._format_duration(eta))

        metrics = Table.grid(expand=True)
        metrics.add_column(justify="left")
        metrics.add_column(justify="right")
        for key in ["loss", "l_diff", "l_x0", "l_ret", "lr"]:
            metrics.add_row(key, self._fmt(state.get(key)))

        throughput = Table.grid(expand=True)
        throughput.add_column(justify="left")
        throughput.add_column(justify="right")
        throughput.add_row("iter_time", f"{self._fmt(state.get('iter_time'), 3)} s")
        throughput.add_row("data_time", f"{self._fmt(state.get('data_time'), 3)} s")
        throughput.add_row("compute_time", f"{self._fmt(state.get('compute_time'), 3)} s")
        throughput.add_row("samples/s", self._fmt(state.get("samples_per_sec"), 2))
        throughput.add_row("wait_ratio", self._fmt(state.get("data_wait_ratio"), 2))

        resources = Table.grid(expand=True)
        resources.add_column(justify="left")
        resources.add_column(justify="right")
        resources.add_row("cpu%", self._fmt(state.get("cpu_percent"), 1))
        resources.add_row("cpu_rss_gb", self._fmt(state.get("cpu_rss_gb"), 2))
        resources.add_row("gpu_alloc(active)", self._fmt(state.get("gpu_allocated_gb"), 2))
        resources.add_row("gpu_resv(cache)", self._fmt(state.get("gpu_reserved_gb"), 2))
        resources.add_row("gpu_peak_resv", self._fmt(state.get("gpu_max_reserved_gb"), 2))

        validation = Table.grid(expand=True)
        validation.add_column(justify="left")
        validation.add_column(justify="right")
        validation.add_row("val_psnr", self._fmt(state.get("val_psnr"), 3))
        validation.add_row("val_ssim", self._fmt(state.get("val_ssim"), 4))
        validation.add_row("val_lpips", self._fmt(state.get("val_lpips"), 4))
        validation.add_row("val_step", self._fmt(state.get("val_step")))

        renderable = Panel(
            Columns(
                [
                    Panel(summary, title="Run", border_style="cyan"),
                    Panel(metrics, title="Train", border_style="magenta"),
                    Panel(throughput, title="Throughput", border_style="green"),
                    Panel(resources, title="Resources", border_style="yellow"),
                    Panel(validation, title="Validation", border_style="blue"),
                ],
                equal=True,
                expand=True,
            ),
            title=self.title,
            border_style="white",
        )
        self.live.update(renderable, refresh=True)

    def stop(self):
        self.live.stop()


def create_training_display(total_steps: int, enabled: bool):
    if not enabled:
        return NullLiveDisplay()

    try:
        if not sys.stdout.isatty():
            return NullLiveDisplay()
        return RichTrainingDisplay(total_steps=total_steps)
    except Exception:
        return NullLiveDisplay()
