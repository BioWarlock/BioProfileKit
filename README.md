# BioProfileKit
[![Python: 3.13](https://img.shields.io/badge/Python-3.13-green.svg)](https://www.python.org/downloads/release/python-3130/) [![License: MIT](https://img.shields.io/badge/License-MIT-darkred.svg)](https://opensource.org/licenses/MIT) 
[![pages-build-deployment](https://github.com/BioWarlock/BioProfileKit/actions/workflows/pages/pages-build-deployment/badge.svg?branch=main)](https://github.com/BioWarlock/BioProfileKit/actions/workflows/pages/pages-build-deployment)


## Overview

BioProfileKit is a specialized bioinformatics tool that enables scientists to analyze large and diverse datasets. 
Unlike traditional profilers, it offers customized analyzes for genomics, proteomics, transcriptomics, and metabolomics, providing sequence analysis and reports on nucleotide or amino acid distribution and abundance. 
It includes advanced visualizations for pattern and anomaly detection, along with interactive dashboards showing key metrics. 
Designed to be user-friendly, BioProfileKit is accessible to scientists without extensive data science skills.
Example Results: [BioProfileKit Example](https://hansen-maria.github.io/BioProfileKit/)

### Features
- **EDA-Focused Analysis** - Offers a detailed overview of data structure, quality, and composition with automated detection of issues like missing values or correlations
- **Specialized Sequence Profiling** - Examines DNA, RNA, and protein sequences using relevant metrics (such as GC content, k-mer frequencies, and amino acid composition)
- **Biological Metadata Recognition** - Automatically identifies and verifies organism names, taxonomic identifiers, and biological annotations using controlled vocabularies from official databases
- **Rich Visualizations** - Creates histograms and interactive charts to help quickly identify patterns
- **Interactive HTML Reports** - Provides portable, user-friendly reports with dynamic filtering and cross-linked visualizations for seamless data exploration

## Installation

```bash
make install
```

**Manual installation**
```bash
pip install -e .
python setup.py build_ext --inplace
```

## Parameters
Currently only supports .csv, .tsv, and .json as input files
```bash 
 bioprofilekit -i input.csv
 Options:
  -i,   --input PATH           Input file as .tsv, .csv or .json  [required]
  -t,   --tax                  Enable taxonomy analysis
  -f,   --func [cog|go]        Choose between COG or GO analysis, if validation is needed
  -tc,  --target_column TEXT   Target column for further analysis
  -k,   --kmer INTEGER         K-mer Size for sequence analysis. Default: 3
  -n,   --top_n INTEGER        Top N entries analysis. Default: 20
  -h,   --help                 Show this message and exit.
```

## Contributing

Contributions to this project are welcome! Whether you find bugs, want to request features, or submit enhancements, please feel free to open an issue or submit a pull request. For major changes, it's recommended to discuss them first to ensure alignment with project goals.
Please read the [`CODE OF CONDUCT`](CODE_OF_CONDUCT.md) to learn more about our guidelines and the contribution process.

## License

Licensed under MIT license ([LICENSE-MIT](LICENSE) or http://opensource.org/licenses/MIT). 
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in BioProfileKit by you, shall be licensed as above, without any additional terms or conditions.

## Contact

For inquiries or support regarding this project, you can reach out to the maintainers through GitHub issues or Discussion.