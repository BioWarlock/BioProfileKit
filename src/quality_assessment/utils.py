def _worst(statuses: list) -> str:
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _rate(value, total):
    return value / total if total else 0.0
