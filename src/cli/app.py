#!/usr/bin/env python3

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
    input_path = Path(input)
    print(colored(f'Reading file {input_path.name}', 'green'))
    df = read_file(input_path) if input_path.suffix != '.parquet' else parse_parquet(input_path)
    general = overview(df, input_path.name)
    print("Starting Multivariate Analysis")
    plots = multivariate_analysis(df, target_column) # ToDo Rename Plots to Multivariate

    dups = df[df.duplicated(keep=False)].reset_index()
    duplicates_table = dups.to_html(
        classes="table table-hover table-responsive nowrap",
        border="0", table_id="dup_table", index=False,
    )

    print(colored(f'Analyse {len(df.columns)} columns', 'blue'))
    column_overviews = [column_overview(df, col) for col in df.columns]

    tax_df = get_tax_ids() if tax else None

    valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_scientific = None, None, None, None, None
    if tax and tax_df is not None:
        valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_scientific = build_lookups(tax_df)

    for col_ov in column_overviews:
        if tax and tax_df is not None:
            col_ov.taxonomy = taxonomy_flags(df, col_ov.name, valid_names, valid_tax_ids, name_to_rank, taxid_to_rank, name_to_scientific)
        if func and tax_df is not None:
            col_ov.annotation = [annotation_flags(df, col_ov.name, func)]
        if hasattr(col_ov, "top_10") and isinstance(col_ov.top_10, pd.Series):
            col_ov.top_10_items = list(col_ov.top_10.items())

        # Biologische Sequenz-Analyse
        if col_ov.sequence == 'dna':
            print(colored(f'Analyzing DNA/RNA sequences in column: {col_ov.name}', 'cyan'))
            col_ov.dna_rna_data = dna_rna_columns(df[col_ov.name], k=kmer, top_n=top_n, invalid=col_ov.invalid_seqs)
        elif col_ov.sequence == 'protein':
            print(colored(f'Analyzing protein sequences in column: {col_ov.name}', 'cyan'))
            col_ov.protein_data = protein_columns(df[col_ov.name], k=kmer, top_n=top_n,  invalid=col_ov.invalid_seqs)
        else:
            col_ov.dna_rna_data = None
            col_ov.protein_data = None

        # Measurement-Analyse
        if col_ov.sequence == 'None':
            measurement_data = measurement_columns(df[col_ov.name], col_ov.name, col_ov.type)
            if measurement_data:
                print(colored(f'Analyzing lab measurements in column: {col_ov.name}', 'cyan'))
            col_ov.measurement_data = measurement_data if measurement_data else None
        else:
            col_ov.measurement_data = None

    print(colored(f'Analyse {len(df.select_dtypes(include="number").columns)} numeric columns', 'blue'))

    exclude_cols = []
    numeric_overviews = [
        numeric_columns(df, col) if not df[col].isnull().all() else exclude_cols.append(col)
        for col in df.select_dtypes(include="number").columns
    ]
    #Todo Identify numeric cat columns
    cat_columns = [
        col for col in df.select_dtypes(include=['str', 'object', 'bool', 'int64', 'float64']).columns
        if any(i.sequence == 'None' for i in column_overviews if i.name == col)
    ]
    print(colored(f'Analyse {len(cat_columns)} object columns', 'blue'))

    categorical_overviews = [
        categorical_columns(df, col) if col not in exclude_cols else None
        for col in cat_columns
    ]
    quality = quality_assessment(general, column_overviews, numeric_overviews,
                                   categorical_overviews, plots)
    print_quality_report(quality)
    output_path = Path(input_path.stem + "_renders")
    print(colored('Writing report …', 'green'))
    write_report(output_path, general, plots, duplicates_table,
                 column_overviews, numeric_overviews, categorical_overviews, top_n, quality)
