import math


def _round1(value):
    return math.floor(value * 10 + 0.5) / 10


def _flag(name, severity, value=None, **extra):
    flag = {"flag": name, "severity": severity}
    if value is not None:
        flag["value"] = value
    flag.update(extra)
    return flag


def _taxonomy_flags(col):
    tax = getattr(col, "taxonomy", None)
    if tax is not None and getattr(tax, "is_taxonomy", False):
        ref = tax.taxid if getattr(tax, "taxid", None) is not None else getattr(tax, "taxonomy", None)
        if ref is None:
            return []
        if isinstance(ref, str):
            return [_flag("Taxonomy Valid", "success")]
        return [_flag("Taxonomy Invalid", "danger", len(ref))]
    if tax is None and getattr(col, "taxonomy_candidate", False):
        return [_flag("Taxonomy Not Checked", "warning")]
    return []


def _annotation_flags(col):
    annotation = getattr(col, "annotation", None)
    anno = annotation[0] if annotation else None
    if not anno or not getattr(anno, "is_annotation", False):
        return []
    if isinstance(anno.valid_annotation, str):
        return [_flag("Annotation Valid", "success")]
    return [_flag("Annotation Invalid", "danger", len(anno.valid_annotation or []))]


def _uniprot_flags(col):
    uniprot = getattr(col, "uniprot", None)
    if not uniprot or not getattr(uniprot, "is_uniprot", False):
        return []
    if uniprot.invalid_values:
        return [_flag("UniProt Invalid", "danger", len(uniprot.invalid_values))]
    return [_flag("UniProt Valid", "success")]


def _missing_flags(col):
    missing = col.missing_per or 0
    if missing == 100:
        return [_flag("Empty Column", "danger", missing)]
    if missing > 15:
        return [_flag("Missing", "danger", missing)]
    if 5 < missing < 15:
        return [_flag("Missing", "warning", missing)]
    if 0.1 < missing < 5:
        return [_flag("Missing", "light", missing)]
    return []


def _correlation_flags(col):
    correlation = getattr(col, "correlation", None)
    if not correlation:
        return []
    high, moderate, low = [], [], []
    for partner, value in ((c[0], c[1]) for c in correlation):
        if abs(value) >= 0.7:
            high.append(partner)
        elif abs(value) >= 0.5:
            moderate.append(partner)
        else:
            low.append(partner)
    flags = []
    if low:
        flags.append(_flag("Low Correlation", "success", partners=low))
    if moderate:
        flags.append(_flag("Moderate Correlation", "warning", partners=moderate))
    if high:
        flags.append(_flag("High Correlation", "danger", partners=high))
    return flags


def _sequence_flags(col):
    if getattr(col, "sequence", "None") == "None":
        return []
    flags = []
    data = getattr(col, "dna_rna_data", None) or getattr(col, "protein_data", None)
    stats = getattr(data, "length_stats", None) if data is not None else None
    if stats is not None and stats.min != stats.max:
        flags.append(_flag("Diff Sequence Length", "secondary", [stats.min, stats.max]))

    dna = getattr(col, "dna_rna_data", None)
    if dna is not None:
        amb = getattr(dna, "ambiguous_base_ratio", None)
        if amb is not None and amb.mean > 0:
            flags.append(_flag("Ambiguous Bases", "danger" if amb.mean >= 5 else "warning", amb.mean))
        rc = getattr(dna, "reverse_complement_ratio", None)
        if rc is not None and rc > 0:
            if rc >= 10:
                flags.append(_flag("RC Duplicates", "danger", rc))
            elif rc >= 1:
                flags.append(_flag("RC Duplicates", "warning", rc))
        low = getattr(dna, "low_complexity", None)
        if low is not None and low.mean < 50:
            flags.append(_flag("Low Complexity", "warning", low.mean))
    return flags


def _categorical_flags(col, cat):
    if col.type not in ("object", "bool", "str") or cat is None:
        return []
    flags = []
    cib = getattr(cat, "cib_ratio", None)
    if cib is not None:
        if cib > 10:
            flags.append(_flag("Imbalanced", "danger", _round1(cib)))
        elif cib > 3:
            flags.append(_flag("Imbalanced", "warning", _round1(cib)))
    rare = getattr(cat, "rare_categories", None)
    if rare is not None and rare > 0:
        flags.append(_flag("Rare Categories", "secondary", rare))
    top1 = getattr(cat, "top_1_coverage", None)
    if top1 is not None and top1 >= 0.95:
        flags.append(_flag("NZV", "danger", _round1(top1 * 100)))
    return flags


def _numeric_flags(col, ov):
    if col.type not in ("int64", "float64") or ov is None:
        return []
    flags = []
    n_total = col.number if col.number and col.number > 0 else 1
    if getattr(ov, "infinity", None) and ov.infinity > 0:
        flags.append(_flag("Infinity", "danger", ov.infinity))
    zero_count = getattr(ov, "zero_count", None)
    if zero_count is not None:
        zero_pct = _round1(zero_count / n_total * 100)
        if zero_pct >= 80:
            flags.append(_flag("Sparse", "danger", zero_pct))
        elif zero_pct >= 50:
            flags.append(_flag("Sparse", "warning", zero_pct))
    out = getattr(ov, "outliers", None)
    if out:
        adjusted = ov.skewness is not None and abs(ov.skewness) > 0.5
        n_out = (out.n_lower_iqr + out.n_upper_iqr) if adjusted else (out.basic_lower_iqr + out.basic_upper_iqr)
        out_pct = _round1(n_out / n_total * 100)
        if out_pct >= 10:
            flags.append(_flag("Outliers", "danger", out_pct))
        elif out_pct >= 5:
            flags.append(_flag("Outliers", "warning", out_pct))
    return flags


def _skew_flags(ov):
    skew = getattr(ov, "skewness", None) if ov is not None else None
    if skew is None:
        return []
    if abs(skew) > 1:
        return [_flag("Highly skewed", "warning", skew)]
    if abs(skew) > 0.5:
        return [_flag("Moderate skew", "secondary", skew)]
    return []


def column_flags(col, ov=None, cat=None):
    flags = _taxonomy_flags(col) + _annotation_flags(col) + _uniprot_flags(col)
    if col.unique == 1:
        flags.append(_flag("Constant", "danger"))
    flags += _missing_flags(col)
    flags += _correlation_flags(col)
    if getattr(col, "monotonicity", None):
        flags.append(_flag("Monotonic", "secondary"))
    flags += _skew_flags(ov)
    flags += _sequence_flags(col)
    measurement = getattr(col, "measurement_data", None)
    if measurement and measurement.unit_counts and len(measurement.unit_counts) > 1:
        flags.append(_flag("Diff Units", "warning", len(measurement.unit_counts)))
    flags += _categorical_flags(col, cat)
    flags += _numeric_flags(col, ov)
    if measurement and measurement.with_measurement:
        flags.append(_flag("Mixed Values", "warning"))
    invalid = getattr(col, "invalid_seqs", None)
    if getattr(col, "suspect_values", None) is not None and not (invalid and len(invalid) > 0):
        flags.append(_flag("Data Anomalies", "warning"))
    if invalid and len(invalid) > 0:
        flags.append(_flag("Invalid Sequences", "danger", len(invalid)))
    return flags


def dataset_flags(general):
    flags = []
    if general.dup_row > 0:
        flags.append(_flag("Duplicate rows", "warning", general.dup_row))
    if general.dup_col > 0:
        flags.append(_flag("Duplicate columns", "warning", general.dup_col))
    if getattr(general, "n_empty", 0):
        flags.append(_flag("Empty columns", "warning", general.n_empty))
    return flags
