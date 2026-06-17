from models import QualityCheck
from .utils import _worst

AMBIGUOUS_SEQ_WARN = 5.0
AMBIGUOUS_SEQ_FAIL = 15.0
STOP_CODON_WARN = 1.0
RC_REDUNDANCY_WARN = 10.0
RC_REDUNDANCY_FAIL = 30.0


def _check_sequence_validity(column_overviews) -> QualityCheck:
    seq_cols = [c for c in column_overviews if c.sequence in ("dna", "protein")]
    if not seq_cols:
        return QualityCheck(name="Sequence Validity", status="pass",
                            message="No sequence columns present", detail_link=None)
    statuses, notes = [], []
    for c in seq_cols:
        n_invalid = len(c.invalid_seqs) if c.invalid_seqs else 0
        if n_invalid > 0:
            statuses.append("warn")
            notes.append(f"{c.name}: {n_invalid} invalid")
        data = getattr(c, "dna_rna_data", None) or getattr(c, "protein_data", None)
        if data is None:
            continue
        amb = getattr(data, "ambiguous_base_ratio", None) or getattr(data, "ambiguous_residue_ratio", None)
        if amb is not None:
            if amb.mean > AMBIGUOUS_SEQ_FAIL:
                statuses.append("fail")
                notes.append(f"{c.name}: {amb.mean:.1f}% mean ambiguous")
            elif amb.mean > AMBIGUOUS_SEQ_WARN:
                statuses.append("warn")
                notes.append(f"{c.name}: {amb.mean:.1f}% mean ambiguous")
        stop = getattr(data, "stop_codon_ratio", None)
        if stop is not None and stop > STOP_CODON_WARN:
            statuses.append("warn")
            notes.append(f"{c.name}: {stop:.1f}% with stop codons")
    status = _worst(statuses) if statuses else "pass"
    return QualityCheck(name="Sequence Validity", status=status,
                        message="; ".join(notes) if notes else "All sequences valid",
                        detail_link="#sequences")


def _check_sequence_redundancy(column_overviews) -> QualityCheck:
    dna_cols = [c for c in column_overviews if c.sequence == "dna"]
    if not dna_cols:
        return QualityCheck(name="Sequence Redundancy (Reverse Complement)", status="pass",
                            message="No DNA/RNA columns present", detail_link=None)
    statuses, notes = [], []
    for c in dna_cols:
        data = getattr(c, "dna_rna_data", None)
        if data is None:
            continue
        rc = getattr(data, "reverse_complement_ratio", None)
        if rc is None:
            continue
        if rc > RC_REDUNDANCY_FAIL:
            statuses.append("fail")
            notes.append(f"{c.name}: {rc:.1f}% RC duplicates")
        elif rc > RC_REDUNDANCY_WARN:
            statuses.append("warn")
            notes.append(f"{c.name}: {rc:.1f}% RC duplicates")
    status = _worst(statuses) if statuses else "pass"
    return QualityCheck(name="Sequence Redundancy (Reverse Complement)", status=status,
                        message="; ".join(notes) if notes else "No significant reverse-complement redundancy",
                        detail_link="#sequences")


def _check_taxonomy_validity(column_overviews) -> QualityCheck:
    tax_cols = [c for c in column_overviews if c.taxonomy is not None]
    if not tax_cols:
        return QualityCheck(name="Taxonomy Validity", status="pass",
                            message="No taxonomy columns present", detail_link=None)
    statuses, notes = [], []
    for c in tax_cols:
        n_invalid = len(c.taxonomy.invalid_names) if c.taxonomy.invalid_names else 0
        if n_invalid > 0:
            statuses.append("warn")
            notes.append(f"{c.name}: {n_invalid} invalid")

    status = _worst(statuses) if statuses else "pass"
    return QualityCheck(name="Taxonomy Validity", status=status,
                        message="; ".join(notes) if notes else "Taxonomy valid",
                        detail_link="#taxonomy")


def _check_unit_validity(column_overviews) -> QualityCheck:
    unit_cols = [c for c in column_overviews if c.measurement_data is not None]
    if not unit_cols:
        return QualityCheck(name="Unit Validity", status="pass",
                            message="No unit columns present", detail_link=None)
    statuses, notes = [], []
    for c in unit_cols:
        ucount = len(c.measurement_data.unit_counts) if c.measurement_data.unit_counts else 0
        if ucount > 1:
            statuses.append("warn")
            notes.append(f"{c.name}: {ucount} units found.")
        if c.measurement_data.with_measurement:
            statuses.append("warn")
            notes.append(f"{c.name}: measurement with unit found.")

    status = _worst(statuses) if statuses else "pass"
    return QualityCheck(name="Unit Validity", status=status,
                        message="; ".join(notes) if notes else "Unit valid",
                        detail_link="#unit")
