import json
import math
from datetime import datetime, timezone

import numpy as np

SCHEMA_VERSION = "1.0"
TOOL_VERSION = "0.1.0"
CORR_THRESHOLD = 0.9 #ToDo check consistencies

def _num(v):
    if v is None:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (int, bool, str)):
        return v
    if isinstance(v, np.ndarray):
        return [_num(x) for x in v.tolist()]
    return v


def _present(v):
    if v is None:
        return False
    if isinstance(v, np.ndarray):
        return v.size > 0
    if hasattr(v, "empty"):          # pandas Series / DataFrame
        return not v.empty
    if isinstance(v, (list, tuple, set, dict, str)):
        return len(v) > 0
    return bool(v)


def _listify(v):
    if not _present(v):
        return None
    if isinstance(v, np.ndarray):
        return [_num(x) for x in v.tolist()]
    if hasattr(v, "tolist"):         # pandas Series
        return [_num(x) for x in v.tolist()]
    if isinstance(v, (list, tuple, set)):
        return [_num(x) for x in v]
    return _num(v)


def _summary(s):
    if s is None:
        return None
    return {"min": _num(s.min), "max": _num(s.max), "mean": _num(s.mean)}


def _outliers(o):
    if o is None:
        return None
    return {
        "lower_bound": _num(o.lower_bound),
        "upper_bound": _num(o.upper_bound),
        "medcouple": _num(o.medcouple),
        "n_lower_iqr": _num(o.n_lower_iqr),
        "n_upper_iqr": _num(o.n_upper_iqr),
        "n_lower_mzscore": _num(o.n_lower_mzscore),
        "n_upper_mzscore": _num(o.n_upper_mzscore),
        "n_lower_zscore": _num(o.n_lower_zscore),
        "n_upper_zscore": _num(o.n_upper_zscore),
    }

def _numeric_block(n):
    if n is None:
        return None
    return {
        "min": _num(n.min), "max": _num(n.max), "mean": _num(n.mean),
        "median": _num(n.median), "mode": _num(n.mode), "std": _num(n.std),
        "sum": _num(n.sum), "kurtosis": _num(n.kurtosis), "skewness": _num(n.skewness),
        "coefficient_of_variation": _num(n.coefficient_of_variation), "mad": _num(n.mad),
        "quantiles": _num(n.quantiles),
        "infinity": _num(n.infinity), "negative_count": _num(n.negative_count),
        "zero_count": _num(n.zero_count),
        "outliers": _outliers(n.outliers),
    }


def _categorical_block(c):
    if c is None:
        return None
    return {
        "unique_categories": _num(c.unique_categories), "mode": c.mode,
        "entropy": _num(c.entropy), "gini": _num(c.gini),
        "simpson_diversity": _num(c.simpson_diversity),
        "max_category_length": _num(c.max_category_length),
        "min_category_length": _num(c.min_category_length),
        "cardinality_ratio": _num(c.cardinality_ratio),
        "rare_categories": _num(c.rare_categories),
        "top_5_coverage": _num(c.top_5_coverage),
        "top_1_coverage": _num(c.top_1_coverage),
        "cib_ratio": _num(c.cib_ratio),
        "effective_cardinality": _num(c.effective_cardinality),
    }


def _dna_block(d):
    return {
        "gc_content": _summary(d.gc_content),
        "ambiguous_base_ratio": _summary(d.ambiguous_base_ratio),
        "length_stats": _summary(d.length_stats),
        "length_outliers": _outliers(d.length_outliers),
        "codon_completeness": _summary(d.codon_completeness),
        "gc_skew": _summary(d.gc_skew), "at_skew": _summary(d.at_skew),
        "cpg_observed_expected": _summary(d.cpg_observed_expected),
        "tpa_observed_expected": _summary(d.tpa_observed_expected),
        "low_complexity": _summary(d.low_complexity),
        "reverse_complement_ratio": _num(d.reverse_complement_ratio),
    }


def _protein_block(p):
    return {
        "ambiguous_residue_ratio": _summary(p.ambiguous_residue_ratio),
        "length_stats": _summary(p.length_stats),
        "length_outliers": _outliers(p.length_outliers),
        "stop_codon_ratio": _num(p.stop_codon_ratio),
        "low_complexity": _summary(p.low_complexity),
        "gravy": _summary(p.gravy),
        "cysteine_count": _summary(p.cysteine_count),
        "disorder_propensity": _summary(p.disorder_propensity),
        "aa_group_distribution": {k: _num(v) for k, v in p.aa_group_distribution.items()}
            if p.aa_group_distribution else None,
    }


def _taxonomy_block(tax):
    if tax is None or not getattr(tax, "is_taxonomy", False):
        return None
    block = {"is_taxonomy": True, "is_mixed": bool(getattr(tax, "is_mixed", False))}
    if getattr(tax, "rank_distribution", None):
        block["rank_distribution"] = {k: _num(v) for k, v in tax.rank_distribution.items()}
    if getattr(tax, "outdated_names", None):
        block["name_corrections"] = tax.outdated_names
    if getattr(tax, "invalid_names", None):
        block["invalid_values"] = tax.invalid_names
    return block


def _measurement_block(m):
    if m is None:
        return None
    return {
        "units": m.units,
        "unit_counts": {str(k): _num(v) for k, v in m.unit_counts.items()},
        "with_measurement": bool(m.with_measurement),
    }


def _correlated_partners(col_name, top_associations, threshold):
    if not top_associations:
        return []
    best = {}
    for p in top_associations:
        if p["value"] < threshold:
            continue
        if p["var1"] == col_name:
            partner = p["var2"]
        elif p["var2"] == col_name:
            partner = p["var1"]
        else:
            continue
        if partner not in best or p["value"] > best[partner]["value"]:
            best[partner] = {"column": partner, "value": _num(p["value"]), "method": p["method"]}
    return sorted(best.values(), key=lambda d: d["value"], reverse=True)


def _role(col_ov, is_empty):
    if is_empty:
        return "empty"
    if col_ov.sequence == "dna":
        return "dna_rna_sequence"
    if col_ov.sequence == "protein":
        return "protein_sequence"
    tax = getattr(col_ov, "taxonomy", None)
    if tax is not None and getattr(tax, "is_taxonomy", False):
        return "taxonomy"
    if getattr(col_ov, "measurement_data", None) is not None:
        return "measurement"
    return None


def _column_entry(col_ov, num, cat, multivariate, is_empty):
    name = col_ov.name
    entry = {"name": name, "dtype": col_ov.type}

    role = _role(col_ov, is_empty)
    if role is not None:
        entry["role"] = role

    if is_empty:
        entry["empty"] = True
        return entry

    entry["general"] = {
        "count": _num(col_ov.number),
        "n_unique": _num(col_ov.unique),
        "missing": _num(col_ov.missing),
        "missing_rate": _num(round((col_ov.missing_per or 0.0) / 100.0, 4)),
        "density": _num(col_ov.density),
        "constant": bool(col_ov.constant) if col_ov.constant is not None else None,
        "mixed_types": _listify(col_ov.mixed_types),
        "suspect_values": _listify(col_ov.suspect_values),
        "monotonicity": bool(col_ov.monotonicity) if col_ov.monotonicity is not None else None,
        "cardinality_dimension_ratio": _num(col_ov.cardinality_dimension_ratio),
        "invalid_seqs": _listify(col_ov.invalid_seqs),
    }

    nb = _numeric_block(num)
    if nb is not None:
        entry["numeric"] = nb
    cb = _categorical_block(cat)
    if cb is not None:
        entry["categorical"] = cb

    # sequence block
    pdata = getattr(col_ov, "protein_data", None)
    ddata = getattr(col_ov, "dna_rna_data", None)
    if pdata is not None:
        entry["sequence"] = _protein_block(pdata)
    elif ddata is not None:
        entry["sequence"] = _dna_block(ddata)

    tb = _taxonomy_block(getattr(col_ov, "taxonomy", None))
    if tb is not None:
        entry["taxonomy"] = tb

    mb = _measurement_block(getattr(col_ov, "measurement_data", None))
    if mb is not None:
        entry["measurement"] = mb

    # multivariate relations attached per column
    target = {}
    ftc = getattr(multivariate, "feature_target_correlation", None)
    if ftc and name in ftc:
        target["association"] = _num(ftc[name]["value"])
        target["method"] = ftc[name]["method"]
    mi = getattr(multivariate, "mutual_information", None)
    if mi and name in mi:
        target["mutual_information"] = _num(mi[name]["value"])
    if target:
        entry["target_relation"] = target

    partners = _correlated_partners(name, getattr(multivariate, "top_associations", None), CORR_THRESHOLD)
    if partners:
        entry["correlated_with"] = partners

    return entry

def _quality_block(quality):
    if quality is None:
        return None
    return {
        "overall": quality.overall,
        "passed": quality.passed, "warnings": quality.warnings,
        "failed": quality.failed, "total": quality.total,
        "categories": [
            {
                "name": cat.name, "status": cat.status,
                "checks": [
                    {"name": c.name, "status": c.status, "message": c.message}
                    for c in cat.checks
                ],
            }
            for cat in quality.categories
        ],
    }


def build_result(general, column_overviews, numeric_overviews,
                 categorical_overviews, multivariate, quality, empty_cols,
                 parameters=None) -> dict:
    num_by = {n.name: n for n in numeric_overviews if n is not None}
    cat_by = {c.name: c for c in categorical_overviews if c is not None}
    empty = set(empty_cols)

    columns = [
        _column_entry(col, num_by.get(col.name), cat_by.get(col.name),
                      multivariate, col.name in empty)
        for col in column_overviews
    ]

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "parameters": parameters or {},
            "correlation_threshold": CORR_THRESHOLD,
        },
        "summary": {
            "filename": general.filename,
            "rows": _num(general.rows), "cols": _num(general.cols),
            "nulls": _num(general.nulls), "nulls_percentage": _num(general.nulls_percentage),
            "empty_rows": _num(general.empty_rows),
            "duplicate_rows": _num(general.dup_row), "duplicate_cols": _num(general.dup_col),
            "rows_per_feature": _num(general.ratio),
            "memory_bytes": _num(general.memory),
            "column_types": {
                "number": _num(general.n_number), "dna": _num(general.n_dna),
                "rna": _num(general.n_rna), "protein": _num(general.n_protein),
                "taxonomy": _num(general.n_taxonomy), "unit": _num(general.n_unit),
                "functional": _num(general.n_functional),
                "categorical": _num(general.n_categorical), "empty": _num(general.n_empty),
            },
        },
        "target": getattr(multivariate, "target_name", None),
        "columns": columns,
    }

def _json_default(o):
    """Last-resort converter for anything that slipped through un-cleaned, so
    json.dump never crashes on a Series/ndarray/numpy scalar."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)

def write_result_json(path, general, column_overviews, numeric_overviews,
                      categorical_overviews, multivariate, quality, empty_cols,
                      parameters=None):
    result = build_result(general, column_overviews, numeric_overviews,
                          categorical_overviews, multivariate, quality, empty_cols,
                          parameters)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
    return path