import re
from enum import Enum

class Sequence(Enum):
    DNA = re.compile('^[acgtn]*$', re.I)
    RNA = re.compile('^[acgun]*$', re.I)
    PROTEIN = re.compile('^[acdefghiklmnpqrstvwyxju]*$', re.I)