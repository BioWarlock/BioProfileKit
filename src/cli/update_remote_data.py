from pathlib import Path

import click
import time

from termcolor import colored

import data_utils.remote_data
from cli.utils import print_step, _fmt_duration
from data_utils.remote_data import get_tax_ids, get_gene_ontology, get_clusters_of_orthologous_groups, get_uniprot_swissprot_metadata

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--tax", is_flag=True, help="Update NCBI Taxonomy")
@click.option("--go", is_flag=True, help="Update Gene Ontology")
@click.option("--cog", is_flag=True, help="Update COG groups")
@click.option("--uniprot",is_flag=True, help="Update UniProt-Swiss-Prot")
@click.option("--all", 'download_all', is_flag=True, help="Update Taxonomy, GO, COG and SwissProt ('not TrEMBL')")
@click.option('--force-refresh', is_flag=True, help="Update even if cache is still fresh")
def update(tax: bool, go: bool, cog: bool, uniprot:bool, download_all: bool, force_refresh: bool) -> None:
    run_start = time.perf_counter()
    if download_all:
        tax = go = cog = uniprot = True

    if not any([tax, go, cog]):
        click.echo("Nothing selected. Use --tax/ --go/ --cog/ --all")
        return

    print(f"\n{'=' * 90}")
    print(colored("  BioProfileKit — Reference Data Download", "magenta", attrs=["bold"]))
    print(f"{'=' * 90}")

    if tax:
        done = print_step("NCBI Taxonomy")
        tax_df = get_tax_ids(force_refresh=force_refresh)
        done(f"{len(tax_df):,} entries")

    if go:
        done = print_step("Gene Ontology")
        go_df = get_gene_ontology(force_refresh=force_refresh)
        done(f"{len(go_df):,} GO terms")

    if cog:
        done = print_step("COG groups")
        cog_df = get_clusters_of_orthologous_groups(force_refresh=force_refresh)
        done(f"{len(cog_df):,} COG groups")

    if uniprot:
        done = print_step("UniProt Swiss-Prot")
        uniprot_df = get_uniprot_swissprot_metadata(force_refresh=force_refresh)
        done(f"{len(uniprot_df):,} UniProt Swiss-Prot entries")

    total = _fmt_duration(time.perf_counter() - run_start)

    output_dir = Path(data_utils.remote_data.BPK_CACHE_ROOT).resolve()

    print(f"\n{'=' * 90}")
    print(colored(f"  Download completed in {total} → {output_dir}", "green", attrs=["bold"]))
    print(f"{'=' * 90}\n")


if __name__=='__main__':
    update()