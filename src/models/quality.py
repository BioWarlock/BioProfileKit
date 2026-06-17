from dataclasses import dataclass


@dataclass
class QualityCheck:
    name: str
    status: str  # pass | warn | fail
    message: str
    detail_link: str | None


@dataclass
class QualityCategory:
    name: str
    status: str
    checks: list  # List[QualityCheck]


@dataclass
class QualityAssessment:
    categories: list  # List[QualityCategory]
    passed: int
    warnings: int
    failed: int
    total: int
    overall: str
