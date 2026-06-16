# BioProfileKit
[![Python: 3.13](https://img.shields.io/badge/Python-3.13-green.svg)](https://www.python.org/downloads/release/python-3130/) [![License: MIT](https://img.shields.io/badge/License-MIT-darkred.svg)](https://opensource.org/licenses/MIT)
[![pages-build-deployment](https://github.com/BioWarlock/BioProfileKit/actions/workflows/pages/pages-build-deployment/badge.svg?branch=main)](https://github.com/BioWarlock/BioProfileKit/actions/workflows/pages/pages-build-deployment)


## Overview

BioProfileKit is a domain-specific bioinformatics profiling tool for tabular biological data. It generates self-contained, interactive HTML reports covering exploratory data analysis, biological sequence profiling, and metadata validation — designed to be accessible to scientists without extensive data science skills.

Example report: [BioProfileKit Example](https://hansen-maria.github.io/BioProfileKit/)

### Features

**Column-level EDA**
- Automated column type inference (numeric, categorical, sequence, metadata)
- Distributional metrics and quality flags per column type
- Numeric quality metrics: IQR, outlier detection (adjusted IQR, modified Z-score, standard Z-score), zero/negative/infinity counts
- Categorical imbalance metrics: class imbalance ratio, rare category detection, near-zero variance, top-1/top-5 coverage
- Collapsible filter bar with text search and quality flag toggles (AND/OR logic)

**Dataset-level EDA**
- Correlation and pairwise structure analysis
- Duplicate detection
- Missing value pattern analysis: MCAR (Little's test), MAR (point-biserial correlation), MNAR (KS-test heuristic)
- Interactive Plotly visualizations

**Biological Sequence Profiling**
- DNA/RNA: GC content, nucleotide distribution, k-mer frequencies, AT/GC skew, ambiguous base ratio, dinucleotide O/E ratios, codon completeness, low-complexity entropy, reverse complement duplicate detection
- Protein: amino acid composition and distribution
- Configurable k-mer size and top-N reporting

**Biological Metadata Validation**
- Organism name and taxonomic identifier verification (Cython-accelerated)
- Functional annotation validation: GO terms and COG categories
- Controlled vocabulary checks against official databases

**Interactive HTML Reports**
- Portable, single-file reports with no server dependency
- Dynamic column filtering with quality flag toggles
- Colorblind-safe theme (Okabe-Ito palette) with persistent toggle
- WCAG AA accessible

## Project Structure

```
BioProfileKit/
├── src/
│   ├── analysis/        # Column-level and dataset-level EDA
│   ├── biological/      # Sequence profiling and metadata validation
│   ├── cli/             # Entry point and report writer
│   ├── cython_wrapper/  # Cython-accelerated components
│   ├── data_utils/      # File I/O and remote data access
│   ├── models/          # Data models per column type
│   └── templates/       # Jinja templates for reports
└── tests/               # Unit and integration tests
```

## Web Application

https://bioprofilekit.computational.bio/


## Installation

```bash
make install
```

**Manual installation**
```bash
pip install -e .
python setup.py build_ext --inplace
```


## Usage

Supports `.csv`, `.tsv`, and `.json` as input formats.

```bash
bioprofilekit -i input.csv

Options:
  -i,  --input PATH           Input file (.tsv, .csv, or .json)  [required]
  -t,  --tax                  Enable taxonomy analysis
  -f,  --func [cog|go]        Functional annotation validation (COG or GO)
  -tc, --target_column TEXT   Target column for further analysis
  -k,  --kmer INTEGER         K-mer size for sequence analysis  [default: 3]
  -n,  --top_n INTEGER        Top N entries for reporting  [default: 20]
  -h,  --help                 Show this message and exit.
```

> [!WARNING]  
> For high-dimensional datasets with many columns, report rendering times may increase significantly. In such cases, the hosted web application at https://bioprofilekit.computational.bio/ is recommended.

## Contributing

Contributions are welcome. For bug reports, feature requests, or enhancements, please open an issue or submit a pull request. For major changes, open an issue first to discuss alignment with project goals.

Please read the [`CODE OF CONDUCT`](CODE_OF_CONDUCT.md) for contribution guidelines.


## License

Licensed under the MIT license ([LICENSE](LICENSE) or https://opensource.org/licenses/MIT).
Unless explicitly stated otherwise, any contribution intentionally submitted for inclusion in BioProfileKit shall be licensed under the same terms.


## Contact

For inquiries or support, reach out via GitHub Issues or Discussions.
