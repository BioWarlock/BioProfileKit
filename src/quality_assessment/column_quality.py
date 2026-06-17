from quality_assessment.utils import _worst, _rate
from models import QualityCheck

MISSING_COL_WARN = 0.30
MISSING_COL_FAIL = 0.50

QUASI_CONSTANT_WARN = 0.95
HIGH_CARDINALITY_WARN = 0.90

SKEW_WARN = 1.0
SKEW_FAIL = 2.0

OUTLIER_WARN = 0.05
OUTLIER_FAIL = 0.15

def _missing_rate(col):
    mp = col.missing_per or 0.0
    return mp / 100.0 if mp > 1 else mp


def _check_missing(column_overviews) -> QualityCheck:
    flagged = []
    worst = "pass"
    for c in column_overviews:
        rate = _missing_rate(c)
        if rate > MISSING_COL_FAIL:
            flagged.append((c.name, rate)); worst = "fail"
        elif rate > MISSING_COL_WARN:
            flagged.append((c.name, rate))
            worst = "fail" if worst == "fail" else "warn"
    if not flagged:
        return QualityCheck(name="Missing Values", status="pass",
                            message="No column above 30% missing", detail_link="#multivariate")
    flagged.sort(key=lambda x: x[1], reverse=True)
    listed = ", ".join(f"{n} ({r * 100:.0f}%)" for n, r in flagged)
    return QualityCheck(name="Missing Values", status=worst,
                        message=f"Columns above 30% missing: {listed}", detail_link="#multivariate")


def _check_variance(column_overviews, categorical_overviews) -> QualityCheck:
    constant_cols = [c.name for c in column_overviews if c.constant]
    quasi = [c.name for c in categorical_overviews
             if c is not None and c.top_1_coverage is not None and c.top_1_coverage >= QUASI_CONSTANT_WARN]
    if constant_cols:
        return QualityCheck(name="Feature Variance", status="fail",
                            message=f"Constant column(s): {', '.join(constant_cols)}", detail_link="#columns")
    if quasi:
        return QualityCheck(name="Feature Variance", status="warn",
                            message=f"Quasi-constant column(s) (top category ≥{QUASI_CONSTANT_WARN:.0%}): {', '.join(quasi)}",
                            detail_link="#columns")
    return QualityCheck(name="Feature Variance", status="pass",
                        message="No constant or quasi-constant columns", detail_link="#columns")


def _check_high_cardinality(column_overviews) -> QualityCheck:
    flagged = [c.name for c in column_overviews
               if c.cardinality_dimension_ratio is not None and c.cardinality_dimension_ratio >= HIGH_CARDINALITY_WARN]
    status = "warn" if flagged else "pass"
    return QualityCheck(
        name="High Cardinality", status=status,
        message=(f"High-cardinality column(s) (≥{HIGH_CARDINALITY_WARN:.0%} unique): {', '.join(flagged)}"
                 if flagged else "No high-cardinality columns"),
        detail_link="#columns",
    )


def _check_mixed_types(column_overviews) -> QualityCheck:
    flagged = [c.name for c in column_overviews if c.mixed_types]
    status = "fail" if flagged else "pass"
    return QualityCheck(
        name="Mixed Types", status=status,
        message=(f"Mixed-type column(s): {', '.join(flagged)}" if flagged else "No mixed-type columns"),
        detail_link="#columns",
    )


def _check_infinity(numeric_overviews) -> QualityCheck:
    total_inf = sum(int(n.infinity) for n in numeric_overviews if n.infinity)
    status = "fail" if total_inf > 0 else "pass"
    return QualityCheck(name="Infinity Check", status=status,
                        message=(f"{total_inf} infinite value(s)" if total_inf else "No infinite values"),
                        detail_link="#columns")


def _check_skewness(numeric_overviews) -> QualityCheck:
    statuses, notes = [], []
    for n in numeric_overviews:
        s = abs(n.skewness) if n.skewness == n.skewness else 0.0  # guard NaN
        if s > SKEW_FAIL:
            statuses.append("fail")
            notes.append(f"{n.name} ({n.skewness:.1f})")
        elif s > SKEW_WARN:
            statuses.append("warn")
            notes.append(f"{n.name} ({n.skewness:.1f})")
    status = _worst(statuses) if statuses else "pass"
    return QualityCheck(
        name="Skewness", status=status,
        message=(f"Strongly skewed (transform recommended): {', '.join(notes)}"
                 if notes else "No strongly skewed numeric columns"),
        detail_link="#columns",
    )


def _check_outliers(numeric_overviews, n_rows) -> QualityCheck:
    statuses, notes = [], []
    for n in numeric_overviews:
        o = n.outliers
        if o is None:
            continue
        flagged = (o.n_lower_iqr or 0) + (o.n_upper_iqr or 0)  # adjusted IQR
        rate = _rate(flagged, n_rows)
        if rate > OUTLIER_FAIL:
            statuses.append("fail")
            notes.append(f"{n.name} ({rate * 100:.1f}%)")
        elif rate > OUTLIER_WARN:
            statuses.append("warn")
            notes.append(f"{n.name} ({rate * 100:.1f}%)")
    status = _worst(statuses) if statuses else "pass"
    return QualityCheck(
        name="Outliers (Adjusted IQR)", status=status,
        message=(f"High outlier share: {', '.join(notes)}" if notes
                 else "No columns with excessive outliers"),
        detail_link="#columns",
    )
