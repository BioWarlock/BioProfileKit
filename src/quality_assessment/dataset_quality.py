from models import QualityCheck
from quality_assessment.utils import _rate

MIN_ROWS_PER_FEATURE_WARN = 10
MIN_ROWS_PER_FEATURE_FAIL = 3

DUP_ROW_WARN = 0.05
DUP_ROW_FAIL = 0.20
EMPTY_ROW_WARN = 0.01

MISSING_COL_WARN = 0.30
MISSING_COL_FAIL = 0.50


def _check_sample_size(general) -> QualityCheck:
    ratio = general.ratio
    if ratio < MIN_ROWS_PER_FEATURE_FAIL:
        status = "fail"
    elif ratio < MIN_ROWS_PER_FEATURE_WARN:
        status = "warn"
    else:
        status = "pass"
    return QualityCheck(
        name="Sample Size vs Feature Count", status=status,
        message=f"{ratio:.1f} rows per feature "
                f"({general.rows} rows / {general.cols} columns)",
        detail_link="#overview",
    )


def _check_duplicate_rows(general) -> QualityCheck:
    dup = general.dup_row
    rate = _rate(dup, general.rows)
    if rate > DUP_ROW_FAIL:
        status = "fail"
    elif rate > DUP_ROW_WARN:
        status = "warn"
    else:
        status = "pass"
    unique_rows = general.rows - dup
    return QualityCheck(
        name="Duplicate Rows", status=status,
        message=(f"{dup} duplicate rows ({rate * 100:.1f}%); "
                 f"{unique_rows} unique of {general.rows}" if dup else "No duplicate rows"),
        detail_link="#overview",
    )


def _check_duplicate_columns(general) -> QualityCheck:
    dup = general.dup_col
    status = "warn" if dup > 0 else "pass"
    return QualityCheck(
        name="Duplicate Columns", status=status,
        message=(f"{dup} duplicate column(s)" if dup else "No duplicate columns"),
        detail_link="#overview",
    )


def _check_empty_rows(general) -> QualityCheck:
    empty = general.empty_rows
    status = "warn" if empty > 0 else "pass"
    return QualityCheck(
        name="Empty Rows", status=status,
        message=(f"{empty} completely empty row(s)" if empty else "No empty rows"),
        detail_link="#overview",
    )
