from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .outliers import Outliers


@dataclass
class SequenceMetricSummary:
    min: float
    max: float
    mean: float

@dataclass
class DNARNAColumns:
    # Per-sequence (Top-20)
    sequence: List[str]
    count: List[int]
    length: List[int]

    # Column-wide summaries
    gc_content: SequenceMetricSummary
    ambiguous_base_ratio: SequenceMetricSummary
    length_stats: SequenceMetricSummary
    length_outliers:Optional[Outliers]
    codon_completeness: SequenceMetricSummary
    gc_skew: SequenceMetricSummary
    at_skew: SequenceMetricSummary
    cpg_observed_expected: SequenceMetricSummary
    tpa_observed_expected: SequenceMetricSummary
    low_complexity: SequenceMetricSummary
    reverse_complement_ratio: float
    reverse_complement_list: set[str]
    # ToDo: Check if needed from Top N or overall
    nucleotide_count: List[Dict[str, int]]
    k_mers: List[List[Tuple[str, int]]]


    # Plots
    plot: str
    gc_distribution: Optional[str]
    length_distribution: Optional[str]
    ambiguous_distribution: Optional[str]
    at_gc_skewness: Optional[str]

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
