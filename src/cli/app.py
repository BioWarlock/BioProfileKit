#!/usr/bin/env python3

import time
from pathlib import Path

import click
import pandas as pd
from termcolor import colored

from analysis.categorical_analysis import categorical_columns
from analysis.multivariate_analysis import multivariate_analysis
from analysis.numeric_analysis import numeric_columns
from analysis.overview import overview, column_overview
from biological.functional_annotation import annotation_flags
from biological.measurement_data import measurement_columns
from biological.sequence_data import dna_rna_columns, protein_columns
from biological.taxonomy import taxonomy_flags, build_lookups
from cli.report_writer import write_report
from cli.utils import _fmt_duration, print_step, info
from data_utils.file_reader import read_file, parse_parquet
from data_utils.remote_data import get_tax_ids
from quality_assessment.quality_assessment import quality_assessment, print_quality_report

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.option("-i", "--input", type=click.Path(exists=True, resolve_path=True), required=True,
              help="Input file as .tsv, .csv or .json")
@click.option('-t', '--tax', is_flag=True, help='Enable taxonomy analysis')
@click.option('-f', '--func', type=click.Choice(['cog', 'go']),
              help='Enable functional annotation analysis. Choose between cog or go')
@click.option('-tc', '--target_column', type=str, help='Target column for Analysis')
@click.option('-k', '--kmer', type=int, default=3, help="K-mer Size for sequence analysis")
@click.option('-n', '--top_n', type=int, default=20, help="Top N entries analysis")
def cli(input: str, tax: bool = False, func: str = None,
        target_column: str = None, kmer: int = None, top_n: int = None):
    run_start = time.perf_counter()
    input_path = Path(input)

    print(f"\n{'=' * 64}")
    print(colored(f"  BioProfileKit — {input_path.name}", "magenta", attrs=["bold"]))
    print(f"{'=' * 64}")

    done = print_step(f"Reading {input_path.name}")
    df = read_file(input_path) if input_path.suffix != '.parquet' else parse_parquet(input_path)
    general = overview(df, input_path.name)
    done(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

    dups = df[df.duplicated(keep=False)].reset_index()
    duplicates_table = dups.to_html(
        classes="table table-hover table-responsive nowrap",
        border="0", table_id="dup_table", index=False,
    )

    done = print_step(f"Column overview ({len(df.columns)} columns)")
    column_overviews = [column_overview(df, col) for col in df.columns]
    done()

    tax_df = get_tax_ids() if tax else None
    valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_scientific = None, None, None, None, None
    if tax and tax_df is not None:
        done = print_step("Building taxonomy lookups")
        valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_scientific = build_lookups(tax_df)
        done(f"{len(valid_names):,} scientific names")

    done = print_step("Per-column analysis")
    seq_count = 0
    for col_ov in column_overviews:
        if tax and tax_df is not None:
            col_ov.taxonomy = taxonomy_flags(df, col_ov.name, valid_names, valid_tax_ids, name_to_rank, taxid_to_rank,  name_to_scientific)

        if func and col_ov.taxonomy is not None:
            col_ov.annotation = [annotation_flags(df, col_ov.name, func)]

        if hasattr(col_ov, "top_10") and isinstance(col_ov.top_10, pd.Series):
            col_ov.top_10_items = list(col_ov.top_10.items())

        if col_ov.sequence == 'dna':
            info(f"DNA sequences: {col_ov.name}")
            col_ov.dna_rna_data = dna_rna_columns(df[col_ov.name], k=kmer, top_n=top_n, invalid=col_ov.invalid_seqs)
            seq_count += 1
        elif col_ov.sequence == 'rna':
            info(f"RNA sequences: {col_ov.name}")
            col_ov.dna_rna_data = dna_rna_columns(df[col_ov.name], k=kmer, top_n=top_n, invalid=col_ov.invalid_seqs)
            seq_count += 1
        elif col_ov.sequence == 'protein':
            info(f"protein sequences: {col_ov.name}")
            col_ov.protein_data = protein_columns(df[col_ov.name], k=kmer, top_n=top_n, invalid=col_ov.invalid_seqs)
            seq_count += 1
        else:
            col_ov.dna_rna_data = None
            col_ov.protein_data = None

        if col_ov.sequence == 'None':
            measurement_data = measurement_columns(df[col_ov.name], col_ov.name, col_ov.type)
            if measurement_data:
                info(f"lab measurements: {col_ov.name}")
            col_ov.measurement_data = measurement_data if measurement_data else None
        else:
            col_ov.measurement_data = None
    done(f"{seq_count} sequence column(s)")

    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    sequence_cols = {col.name for col in column_overviews if col.sequence in ('dna', 'rna', 'protein')}

    numeric_cols = [col for col in df.select_dtypes(include="number").columns
                    if col not in empty_cols and col not in sequence_cols]
    done = print_step(f"Numeric analysis ({len(numeric_cols)} columns)")
    numeric_overviews = [numeric_columns(df, col) for col in numeric_cols]
    done()

    cat_columns = [col for col in df.select_dtypes(include=['str', 'object', 'bool']).columns
                   if col not in empty_cols and col not in sequence_cols]
    done = print_step(f"Categorical analysis ({len(cat_columns)} columns)")
    categorical_overviews = [categorical_columns(df, col) for col in cat_columns]
    done()

    done = print_step("Multivariate analysis")
    plots = multivariate_analysis(df, target_column)
    done()

    done = print_step("Quality assessment")
    quality = quality_assessment(general, column_overviews, numeric_overviews,
                                 categorical_overviews, plots)
    done()
    print_quality_report(quality)

    general.n_number = len(numeric_cols)
    general.n_categorical = len(cat_columns)
    general.n_empty = len(empty_cols)
    general.n_dna = sum(1 for col in column_overviews if col.sequence == 'dna')
    general.n_rna = sum(1 for col in column_overviews if col.sequence == 'rna')
    general.n_protein = sum(1 for col in column_overviews if col.sequence == 'protein')
    general.n_taxonomy = sum(1 for col in column_overviews
                             if getattr(col, 'taxonomy', None) and col.taxonomy.is_taxonomy)
    general.n_unit = sum(1 for col in column_overviews if col.measurement_data is not None)
    general.n_functional = sum(1 for col in column_overviews if getattr(col, 'annotation', None))
    
    output_path = Path(input_path.stem + "_renders")
    done = print_step("Writing report")
    write_report(output_path, general, plots, duplicates_table,
                 column_overviews, numeric_overviews, categorical_overviews, top_n, quality)
    done(f"→ {output_path}/")

    total = _fmt_duration(time.perf_counter() - run_start)
    print(f"\n{'=' * 64}")
    print(colored(f"  Done in {total}", "green", attrs=["bold"]))
    print(f"{'=' * 64}\n")
