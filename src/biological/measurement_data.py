import re
from typing import List

import pandas as pd

from models.measurement import UNITColumns
from models.overview import ColumnOverview
from enums.measurement_enum import MEASUREMENTS


def measurement_columns(column_overview: ColumnOverview, df: pd.DataFrame) -> UNITColumns | bool:
    if column_overview.sequence == 'None':
        name_match = MEASUREMENTS.UNIT_IN_COL_TITLE.value.match(column_overview.name)
        if name_match:
            unit: str = name_match.group(0).lstrip('[').rstrip(']')
            return UNITColumns(
                units=[unit],
                unit_counts=[{unit: 1}],
                with_measurement=False
            )
        if column_overview.type == 'str':
            values = df[column_overview.name].dropna().unique().astype(str).tolist()
            if all(len(x) > 1 for x in values):
                measurement_and_unit = match_units(values, MEASUREMENTS.UNIT_COLUMN.value)
                if len(measurement_and_unit) > 0:
                    return UNITColumns(
                        units=[unit.split(' ')[1] if ' ' in unit else unit for unit in measurement_and_unit],
                        unit_counts=df[column_overview.name].value_counts(dropna=False).to_dict(),
                        with_measurement=any([has_number_and_unit(unit) for unit in measurement_and_unit])
                    )
    return False

def match_units(entries: List[str], regex: re.Pattern) -> List[str]:
    units: List[str] = []
    for entrie in entries:
        measurement_and_unit = regex.fullmatch(entrie)
        if measurement_and_unit:
            units.append(measurement_and_unit.group(0))
    return units

def has_number_and_unit(value: str) -> bool:
    special_units = {'1/s', '1/m', '1/M', '1/h', '1/min'} # Add new ones, if needed
    if value in special_units:
        return False
    pattern = re.compile(r'^-?\d+\.?\d*\s*[a-zA-Z°/%]+$')
    return bool(pattern.match(value) and value not in special_units)
