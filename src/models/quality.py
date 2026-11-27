from dataclasses import dataclass, field

@dataclass
class QualityCheck:
    name: str
    status: str  # pass | warn | fail
    message: str
    detail_link: str | None
    items: list = field(default_factory=list)  # List[dict]: affected columns / pairs


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