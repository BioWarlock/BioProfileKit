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
- Correlation and pairwise structure analysis: Pearson (numeric), Cramér's V (categorical), Eta² (mixed), combined association heatmap
- Duplicate detection with grouped duplicate-row reporting
- Missing value pattern analysis: MCAR (Little's test), MAR (point-biserial correlation), MNAR (KS-test heuristic)
- Target/feature analysis: feature–target association ranking, mutual information (with top-relationship plots), class balance
- Interactive Plotly visualizations

**Biological Sequence Profiling**
- DNA/RNA: GC content, nucleotide distribution, k-mer frequencies, AT/GC skew, ambiguous base ratio, dinucleotide O/E ratios, codon completeness, low-complexity entropy, reverse complement duplicate detection
- Protein: amino acid composition and distribution, physicochemical properties (GRAVY, aliphatic index, charge, isoelectric point, instability index, and more)
- Configurable k-mer size and top-N reporting

**Biological Metadata Validation**
- Organism name and taxonomic identifier verification against NCBI Taxonomy (Cython-accelerated), with taxonomic composition sunburst plots
- UniProt validation against Swiss-Prot or TrEMBL, with per-column match badges
- Functional annotation validation against GO terms or COG categories, with summary plots
- Controlled vocabulary checks against official databases; reference data is cached locally and refreshed on a configurable schedule (see [Reference Data](#reference-data))

**Data Quality Assessment**
- Automated pass/warning/fail checks grouped by category (e.g. missingness, duplicates, sequence validity, taxonomy, functional annotation)
- Overall quality banner plus per-category and per-check summaries, with deep links into the relevant report section

**Interactive HTML Reports**
- Portable, single-file reports with no server dependency
- Dynamic column filtering with quality flag toggles
- Colorblind-safe theme (Okabe-Ito palette) with persistent toggle
- WCAG AA accessible

## Project Structure

```
BioProfileKit/
├── src/
│   ├── analysis/            # Column-level and dataset-level EDA
│   ├── biological/          # Sequence profiling and metadata validation (taxonomy, UniProt, GO/COG)
│   ├── cli/                 # CLI entry points (bioprofilekit, bioprofiledata) and report writer
│   ├── cython_wrapper/      # Cython-accelerated components
│   ├── data_utils/          # File I/O and remote reference-data access/caching
│   ├── enums/               # Shared enums and controlled vocabularies
│   ├── models/              # Data models per column type
│   ├── quality_assessment/  # Automated data-quality checks and report aggregation
│   ├── static/              # Static assets (CSS, JS, images) bundled into generated reports
│   └── templates/           # Jinja templates for reports
└── tests/
    ├── unit/                # Unit tests per module
    └── integration/         # End-to-end report-generation tests
```

## Web Application

BioProfileKit is also available as a hosted web service at [bioprofilekit.computational.bio](https://bioprofilekit.computational.bio/), for users who prefer not to install the CLI locally or who need to profile larger datasets without waiting on local report rendering. It wraps the same CLI core behind a Vue 3/TypeScript frontend and a FastAPI backend, runs analyses as jobs on de.NBI cloud infrastructure (Kubernetes), and authenticates via Keycloak/de.NBI AAI.


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
  -i,  --input PATH                  Input file (.tsv, .csv, or .json)  [required]
  -t,  --tax                         Enable taxonomy analysis
  -u,  --uniprot [swissprot|trembl]  Enable UniProt validation against Swiss-Prot or TrEMBL
  -f,  --func [cog|go]               Functional annotation validation (COG or GO)
  -tc, --target_column TEXT          Target column for further analysis
  -k,  --kmer INTEGER                K-mer size for sequence analysis  [default: 3]
  -n,  --top_n INTEGER               Top N entries for reporting  [default: 20]
  -h,  --help                        Show this message and exit.
```

> [!WARNING]  
> For high-dimensional datasets with many columns, report rendering times may increase significantly. In such cases, the hosted web application at https://bioprofilekit.computational.bio/ is recommended.

## Reference Data

Taxonomy, GO, COG, and UniProt Swiss-Prot/TrEMBL reference data (used by `-t`/`-f`/`-u`) are downloaded on first use and cached locally — by default under `~/.local/share/bioprofilekit`, configurable via the `BPK_CACHE_DIR` environment variable. Each cache is considered fresh for 30 days and re-downloaded automatically once it expires.

To refresh the cache explicitly, ahead of time or outside of a report run:

```bash
bioprofiledata --all              # Taxonomy, GO, COG, and UniProt Swiss-Prot
bioprofiledata --tax --go         # only selected datasets
bioprofiledata --all --force-refresh   # ignore cache freshness

Options:
  --tax             Update NCBI Taxonomy
  --go              Update Gene Ontology
  --cog             Update COG groups
  --uniprot         Update UniProt Swiss-Prot
  --all             Update Taxonomy, GO, COG and Swiss-Prot ("not TrEMBL")
  --force-refresh   Update even if cache is still fresh
  -h, --help        Show this message and exit.
```

## Contributing

Contributions are welcome. For bug reports, feature requests, or enhancements, please open an issue or submit a pull request. For major changes, open an issue first to discuss alignment with project goals.

Please read the [`CODE OF CONDUCT`](CODE_OF_CONDUCT.md) for contribution guidelines.


## License

Licensed under the MIT license ([LICENSE](LICENSE) or https://opensource.org/licenses/MIT).
Unless explicitly stated otherwise, any contribution intentionally submitted for inclusion in BioProfileKit shall be licensed under the same terms.


## Contact

For inquiries or support, reach out via GitHub Issues or Discussions.