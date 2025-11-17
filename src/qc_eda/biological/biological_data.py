import tempfile
import warnings
from collections import defaultdict, Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Tuple
from Bio import motifs
from weblogo import *
import numpy as np
import pandas as pd
import peptides
import plotly.express as px
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

@dataclass
class DNARNAColumns:
    sequence: List[str]
    gc_content: List[float]
    length: List[int]
    count: List[int]
    nucleotide_count: List[Dict[str, int]]
    k_mers: List[List[Tuple[str, int]]]
    plot: str

#ToDo: Add Composition over all
@dataclass
class PROTEINColumns:
    sequence: List[str]
    length: List[int]
    count: List[int]
    composition: List[Dict[str, int]]
    frequency: List[float]
    hydrophobicity: List[float]
    charge: List[float]
    molecular_weight: List[float]
    isoelectric_point: List[float]
    aliphatic_index: List[float]
    boman: List[float]
    aromaticity: List[float]
    instability: List[float]
    k_mers: List[List[Tuple[str, int]]]
    plot: str


def count_nmer(sequence, n) -> defaultdict:
    seqs = np.frombuffer(sequence.encode('utf-8'), dtype='S1')
    nmere = np.lib.stride_tricks.sliding_window_view(seqs, n)
    unique, counts = np.unique(nmere, return_counts=True, axis=0)
    result = defaultdict(int)
    for nmer, count in zip(unique, counts):
        nmer_str = b"".join(nmer).decode('utf-8')
        result[nmer_str] += count
    return result


def top_mere(seq, n=3, top=5) -> List[Tuple[str, int]] | None:
    if not seq or len(seq) < n:
        return None
    counts = count_nmer(seq, n)
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top]


def biological_data_top_entries(seqs: pd.Series, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    arr = np.char.upper(seqs.to_numpy(dtype=str))
    uniq_tmp, counts_tmp = np.unique(arr, return_counts=True)
    
    top_k = min(top_k, len(uniq_tmp))
    top_idx = np.argsort(counts_tmp)[::-1][:top_k]
    
    uniques = uniq_tmp[top_idx]
    counts = counts_tmp[top_idx]

    lengths = np.array([len(s) for s in uniques])
    min_len, max_len = lengths.min(), lengths.max()
    
    return uniques, counts, min_len, max_len, lengths


def dna_rna_columns(seqs: pd.Series, k: int = 3, top_n: int = 20, top: int = 5) -> DNARNAColumns:
    uniques, counts, min_len, max_len, lengths = biological_data_top_entries(seqs, top_n)

    gc_count = np.char.count(uniques, 'G') + np.char.count(uniques, 'C')
    gc_content = np.round(np.where(lengths > 0, gc_count / lengths * 100, 0.0), 2).tolist()
    nucleotide_count = [dict(Counter(seq)) for seq in uniques]

    k_mers = _kmer_check(k, top, uniques)

    if min_len == max_len:
        plot = make_logo(uniques,'color_classic')
    else:
        flat_kmers = chain.from_iterable(k_mers)
        df_kmers = pd.DataFrame(flat_kmers, columns=['kmer', 'count'])
        aggregated = df_kmers.groupby('kmer', as_index=False)['count'].sum()
        plot = plot_overview(aggregated['kmer'].tolist(), aggregated['count'].tolist())

    return DNARNAColumns(
        sequence=uniques.tolist(),
        gc_content=gc_content,
        length=lengths.tolist(),
        count=counts.tolist(),
        nucleotide_count=nucleotide_count,
        k_mers=k_mers,
        plot=plot
    )

def _kmer_check(k: int, top: int, uniques: np.ndarray) -> list:
    check_length = any(len(i) <= k for i in uniques)
    if not check_length:
        k_mers = [top_mere(seq, n=k, top=top) for seq in uniques]
    else:
        warnings.warn("Warning: Sequence length is smaller than choosen k-Mer Size. Setting k-Mer Size to 3.")
        k_mers = [top_mere(seq, n=3, top=top) for seq in uniques]
    return k_mers


def protein_descriptors(peptide: str) -> Dict[str, str | float | dict[str, float]]:
    descriptors: Dict[str, str | float | dict[str, float]] = {}
    p: peptides.Peptide = peptides.Peptide(peptide)
    descriptors["seq"] = peptide
    descriptors["freq"] = p.frequencies()

    try:
        descriptors["aidx"] = p.aliphatic_index()
    except ZeroDivisionError:
        descriptors["aidx"] = 0.0
    descriptors["boman"] = p.boman()
    descriptors["charge"] = p.charge()
    descriptors["hp"] = p.hydrophobicity()
    descriptors["iep"] = p.isoelectric_point()
    descriptors["iidx"] = p.instability_index()
    descriptors["mol"] = p.molecular_weight()
    descriptors["aroma"] = sum([peptide.count(aa) for aa in ('F', 'W', 'Y')]) / len(peptide)
    return descriptors


def protein_columns(seqs: pd.Series,  k: int = 3, top_n: int = 20, top: int = 5) -> PROTEINColumns:
    uniques, counts, min_len, max_len, lengths = biological_data_top_entries(seqs, top_n)

    aa_composition = [dict(Counter(seq)) for seq in uniques]
    descriptors = [protein_descriptors(seq) for seq in uniques]

    k_mers = _kmer_check(k, top, uniques)
    #ToDo Add Desclaimer
    if min_len == max_len:
        plot = make_logo(uniques, "color_chemistry")
    else:
        flat_kmers = chain.from_iterable(k_mers)
        df_kmers = pd.DataFrame(flat_kmers, columns=['kmer', 'count'])
        aggregated = df_kmers.groupby('kmer', as_index=False)['count'].sum()
        plot = plot_overview(aggregated['kmer'].tolist(), aggregated['count'].tolist())


    return PROTEINColumns(
        sequence=uniques.tolist(),
        length=lengths.tolist(),
        count=counts.tolist(),
        composition=aa_composition,
        frequency=[descriptor['freq'] for descriptor in descriptors],
        hydrophobicity=[descriptor['hp'] for descriptor in descriptors],
        charge=[descriptor['charge'] for descriptor in descriptors],
        molecular_weight=[descriptor['mol'] for descriptor in descriptors],
        isoelectric_point=[descriptor['iep'] for descriptor in descriptors],
        aliphatic_index=[descriptor['aidx'] for descriptor in descriptors],
        boman=[descriptor['boman'] for descriptor in descriptors],
        aromaticity=[descriptor['aroma'] for descriptor in descriptors],
        instability=[descriptor['iidx'] for descriptor in descriptors],
        k_mers=k_mers,
        plot=plot
    )


def make_logo(seqs, color):
    m = motifs.create(seqs)
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        m.weblogo(tmp_path, format="svg",color_scheme=color, logo_font="Calibri",logo_margin=3, fontsize=12)

        with open(tmp_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()
            if '<svg ' in svg_content:
                svg_content = svg_content.replace(
                    '<svg ',
                    '<svg style="width: 100%; height: 250px; max-width: 800px;" '
                )

        return svg_content

    finally:
        if Path(tmp_path).is_file():
            Path(tmp_path).unlink(missing_ok=True)


def plot_overview(kmer, count):
    fig = px.bar(x=kmer, y=count, color_discrete_sequence=['#0F65A0'])
    fig.update_layout(bargap=0.2, plot_bgcolor='white', xaxis_title='K-mers',
        yaxis_title='Count')
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        tickangle=-45

    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    # fig.write_image("test.png")
    return fig.to_html(full_html=False, include_plotlyjs='cdn')