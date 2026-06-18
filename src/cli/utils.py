import time

from termcolor import colored


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s"


def print_step(label: str):
    """Print a phase header and return a callable that closes it with timing."""
    print(f"▶ {label} …")
    start = time.perf_counter()

    def done(detail: str = ""):
        elapsed = _fmt_duration(time.perf_counter() - start)
        suffix = f" — {detail}" if detail else ""
        print(colored(f"  ✓", "green"), f"{label} ({elapsed}){suffix}")

    return done


def info(message: str):
    print(colored(f"    {message}", "blue"))
