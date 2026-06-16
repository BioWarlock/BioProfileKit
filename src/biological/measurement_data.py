import re
from typing import List

import pandas as pd

from models.measurement import UNITColumns
from enums.measurement_enum import MEASUREMENTS


def measurement_columns(col: pd.Series, name: str, col_type: str) -> UNITColumns | bool:
    name_match = MEASUREMENTS.UNIT_IN_COL_TITLE.value.match(name)
    if name_match:
        unit: str = name_match.group(0).lstrip('[').rstrip(']')
        return UNITColumns(
            units=[unit],
            unit_counts={unit: 1},
            with_measurement=False
        )
    if col_type == 'str':
        values = col.dropna().unique().astype(str).tolist()
        if all(len(x) > 1 for x in values):
            measurement_and_unit = match_units(values, MEASUREMENTS.UNIT_COLUMN.value)
            if len(measurement_and_unit) == len(values):
                measurement_passed = True
            else:
                all_values = col.dropna().astype(str).tolist()
                all_measurement_and_unit = match_units(values, MEASUREMENTS.UNIT_COLUMN.value)
                measurement_passed = len(all_measurement_and_unit) / len(all_values) > 0.95
            if measurement_passed:
                return UNITColumns(
                    units=[unit.split(' ')[1] if ' ' in unit else unit for unit in measurement_and_unit],
                    unit_counts=col.value_counts().to_dict(),
                    with_measurement=any([has_number_and_unit(unit) for unit in measurement_and_unit])
                )
    return False

def match_units(entries: List[str], regex: re.Pattern) -> List[str]:
    units: List[str] = []
    for entry in entries:
        measurement_and_unit = regex.fullmatch(entry)
        if measurement_and_unit:
            units.append(measurement_and_unit.group(0))
    return units

def has_number_and_unit(value: str) -> bool:
    special_units = {'1/s', '1/m', '1/M', '1/h', '1/min'} # Add new ones, if needed
    if value in special_units:
        return False
    pattern = re.compile(r'^-?\d+\.?\d*\s*[a-zA-Z°/%]+$')
    return bool(pattern.match(value) and value not in special_units)
