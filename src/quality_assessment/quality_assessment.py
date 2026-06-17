from models.quality import QualityCategory, QualityAssessment
from quality_assessment.biological_quality import _check_sequence_validity, _check_sequence_redundancy, \
    _check_taxonomy_validity, _check_unit_validity
from quality_assessment.column_quality import _check_missing, _check_variance, _check_high_cardinality, \
    _check_mixed_types, _check_infinity, _check_skewness, _check_outliers
from quality_assessment.dataset_quality import _check_sample_size, _check_duplicate_rows, _check_duplicate_columns, \
    _check_empty_rows
from quality_assessment.relationships import _check_leakage, _check_multicollinearity
from quality_assessment.utils import _worst


def quality_assessment(general, column_overviews, numeric_overviews, categorical_overviews, multivariate) -> QualityAssessment:
    numeric_overviews = [n for n in numeric_overviews if n is not None]
    categorical_overviews = [c for c in categorical_overviews if c is not None]

    structure = [
        _check_sample_size(general),
        _check_duplicate_rows(general),
        _check_duplicate_columns(general),
        _check_empty_rows(general),
    ]
    column_quality = [
        _check_missing(column_overviews),
        _check_variance(column_overviews, categorical_overviews),
        _check_high_cardinality(column_overviews),
        _check_mixed_types(column_overviews),
        _check_infinity(numeric_overviews),
        _check_skewness(numeric_overviews),
        _check_outliers(numeric_overviews, general.rows),
    ]
    biological = [
        _check_sequence_validity(column_overviews),
        _check_sequence_redundancy(column_overviews),
        _check_taxonomy_validity(column_overviews),
        _check_unit_validity(column_overviews),
    ]
    relationships = [
        _check_leakage(multivariate),
        _check_multicollinearity(multivariate),
    ]

    categories = [
        QualityCategory("Dataset Structure", _worst([c.status for c in structure]), structure),
        QualityCategory("Column Quality", _worst([c.status for c in column_quality]), column_quality),
        QualityCategory("Biological", _worst([c.status for c in biological]), biological),
        QualityCategory("Relationships", _worst([c.status for c in relationships]), relationships),
    ]

    all_checks = [c for cat in categories for c in cat.checks]
    passed = sum(1 for c in all_checks if c.status == "pass")
    warnings = sum(1 for c in all_checks if c.status == "warn")
    failed = sum(1 for c in all_checks if c.status == "fail")

    overall = "not_ready" if failed else ("caution" if warnings else "ready")

    return QualityAssessment(
        categories=categories, passed=passed, warnings=warnings,
        failed=failed, total=len(all_checks), overall=overall,
    )


def print_quality_report(quality: QualityAssessment) -> None:
    symbols = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}
    overall_label = {"ready": "READY", "caution": "READY WITH CAUTION", "not_ready": "NOT READY"}

    print("\n" + "=" * 64)
    print("QUALITY ASSESSMENT".center(64))
    print("=" * 64)
    print(f"  Overall: {overall_label.get(quality.overall, quality.overall)}")
    print(f"  {quality.passed} passed | {quality.warnings} warnings | "
          f"{quality.failed} failed  (of {quality.total})")

    for cat in quality.categories:
        print("-" * 64)
        print(f"  {symbols.get(cat.status)} {cat.name.upper()}")
        for c in cat.checks:
            print(f"     {symbols.get(c.status)} {c.name}")
            print(f"            {c.message}")
    print("=" * 64)
