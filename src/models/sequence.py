from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class DNARNAColumns:
    sequence: List[str]
    gc_content: List[float]
    length: List[int]
    count: List[int]
    nucleotide_count: List[Dict[str, int]]
    k_mers: List[List[Tuple[str, int]]]
    plot: str

# ToDo: Add Composition over all
@dataclass
class PROTEINColumns:
    sequence: List[str]
    length: List[int]
    count: List[int]
    composition: List[Dict[str, int]]
    frequency: List[float]
    hydrophobicity: List[float]
    charge: List[float]
    molecular_weight: List[float]
    isoelectric_point: List[float]
    aliphatic_index: List[float]
    boman: List[float]
    aromaticity: List[float]
    instability: List[float]
    k_mers: List[List[Tuple[str, int]]]
    plot: str
