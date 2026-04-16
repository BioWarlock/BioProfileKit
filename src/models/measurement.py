from dataclasses import dataclass
from typing import List, Dict


@dataclass
class UNITColumns:
    units: List[str]
    unit_counts: List[Dict[str | None, int]]
    with_measurement: bool
