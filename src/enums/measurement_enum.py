import re
from enum import Enum

class MEASUREMENTS(Enum):
    UNIT_IN_COL_TITLE = re.compile(r'\[[a-zA-Zµμ°/]+]', re.I)
    UNIT_PREFIXES = r'(Q|R|Y|Z|E|P|T|G|M|k|h|da|d|c|m|µ|μ|micro|n|p|f|a|z|y|r|q)'
    UNITS = r'(g|mol|M|m|L|l|s|min|unit|units|°C|C|K|A|cd|h|24 h|d|g creatinine|Eq|%|U|kU|IU|L (%)|% of total Hb|Fraction of total Hb|Osm|AU|angstroms|\d)'
    UNIT_COLUMN = re.compile(rf'^((\d\.,)+\s*)?{UNIT_PREFIXES}?{UNITS}(/{UNIT_PREFIXES}?{UNITS})*$', re.I)
