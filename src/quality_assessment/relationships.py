from models import QualityCheck

LEAKAGE_WARN = 0.90
LEAKAGE_FAIL = 0.98
MULTICOLLINEAR_WARN = 0.90


def _check_leakage(multivariate) -> QualityCheck:
    ftc = getattr(multivariate, "feature_target_correlation", None)
    if not ftc:
        return QualityCheck(name="Target Leakage", status="pass",
                            message="No target set; leakage check skipped", detail_link=None)
    suspects = [(f, info["value"]) for f, info in ftc.items() if info["value"] >= LEAKAGE_WARN]
    if not suspects:
        return QualityCheck(name="Target Leakage", status="pass",
                            message="No feature strongly associated with the target",
                            detail_link="#multivariate")
    suspects.sort(key=lambda x: x[1], reverse=True)
    status = "fail" if suspects[0][1] >= LEAKAGE_FAIL else "warn"
    listed = ", ".join(f"{f} ({v:.2f})" for f, v in suspects)
    return QualityCheck(name="Target Leakage", status=status,
                        message=f"Suspiciously high feature-target association: {listed}",
                        detail_link="#multivariate")


def _check_multicollinearity(multivariate) -> QualityCheck:
    pairs = getattr(multivariate, "top_associations", None)
    if not pairs:
        return QualityCheck(name="Multicollinearity", status="pass",
                            message="No strongly associated feature pairs", detail_link="#multivariate")
    high = [p for p in pairs if p["value"] >= MULTICOLLINEAR_WARN]
    if not high:
        return QualityCheck(name="Multicollinearity", status="pass",
                            message="No feature pair above 0.90 association", detail_link="#multivariate")
    listed = ", ".join(f"{p['var1']}↔{p['var2']} ({p['value']:.2f})" for p in high)
    return QualityCheck(name="Multicollinearity", status="warn",
                        message=f"Highly associated feature pairs: {listed}", detail_link="#multivariate")
