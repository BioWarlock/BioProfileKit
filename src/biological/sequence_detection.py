import re
import math

from pandas.api.types import infer_dtype

from enums.sequence_enum import Sequence
from cython_wrapper.wrapper_utils import fast_check_sequence, char_entropy


def _alphabet_from_pattern(pattern):
    match = re.search(r'\[([^]]+)]', pattern.pattern)
    return set(match.group(1).upper()) if match else set()


DNA_ALPHABET = _alphabet_from_pattern(Sequence.DNA.value)
RNA_ALPHABET = _alphabet_from_pattern(Sequence.RNA.value)
PROTEIN_ALPHABET = _alphabet_from_pattern(Sequence.PROTEIN.value)

ENTROPY_THRESHOLDS = {
    "dna": 0.5 * math.log2(len(DNA_ALPHABET)),
    "rna": 0.5 * math.log2(len(RNA_ALPHABET)),
    "protein": 0.5 * math.log2(len(PROTEIN_ALPHABET)),
}


def check_sequence(df, col, threshold=0.92):
    if df[col].name in df.select_dtypes(include=['number', 'bool']).columns or infer_dtype(df[col]).__contains__('mixed'):
        return "None", []
    if df[col].astype(str).str.len().eq(1).all():
        return "None", []
    values = df[col].dropna().astype(str).tolist()

    unique_count = len(set(values))
    if unique_count < 10:
        return "None", []

    unique_values = list(set(values))
    non_alpha_pattern = re.compile(r'[^a-zA-Z]')
    non_alpha_count = sum(1 for v in unique_values if non_alpha_pattern.search(v))
    if non_alpha_count / len(unique_values) > 0.3:
        return "None", []

    if all(len(x) > 2 for x in values):
        match, invalid = fast_check_sequence(values, Sequence.DNA.value, threshold)
        if match:
            return "dna", _get_invalid(values, invalid)
        match, invalid = fast_check_sequence(values, Sequence.RNA.value, threshold)
        if match:
            return "rna", _get_invalid(values, invalid)
        match, invalid = fast_check_sequence(values, Sequence.PROTEIN.value, threshold)
        if match:
            if not invalid or char_entropy(values, PROTEIN_ALPHABET) >= ENTROPY_THRESHOLDS["protein"]:
                return "protein", _get_invalid(values, invalid)
    return "None", []


def _get_invalid(values, invalid_indices):
    if not invalid_indices:
        return []
    return [values[i] for i in invalid_indices]