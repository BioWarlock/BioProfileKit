import re
import tempfile
from collections import defaultdict, Counter
from itertools import chain
from pathlib import Path
from typing import Tuple, List, Dict
from urllib.error import URLError, HTTPError

from Bio import motifs
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import pandas as pd
import peptides
import plotly.express as px
import ssl

from analysis.outlier_detection import detect_outliers
from analysis.plot_utils import apply_standard_axes
from biological.plot_utils import length_distribution, gc_distribution, ambiguous_distribution, at_gc_skewness, \
    aa_group_distribution
from models.sequence import DNARNAColumns, PROTEINColumns, SequenceMetricSummary

ssl._create_default_https_context = ssl._create_stdlib_context


# ToDo: Add Composition over all


def count_nmer(sequence, n) -> defaultdict:
    if n > len(sequence):
        return defaultdict(int)
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


def biological_data_top_entries(seqs: pd.Series, top_k: int = 20) -> Tuple[
    np.ndarray, np.ndarray, int, int, np.ndarray]:
    seqs = seqs.str.upper()
    vc = seqs.value_counts()
    uniq_tmp, counts_tmp = vc.index.to_numpy(), vc.values

    top_k = min(top_k, len(uniq_tmp))
    top_idx = np.argsort(counts_tmp)[::-1][:top_k]

    uniques = uniq_tmp[top_idx].astype(str)
    counts = counts_tmp[top_idx]

    lengths = np.array([len(s) for s in uniques])
    min_len, max_len = lengths.min(), lengths.max()

    return uniques, counts, min_len, max_len, lengths


def dna_rna_columns(seqs: pd.Series, k: int = 3, top_n: int = 20, top: int = 5, invalid: list=None) -> DNARNAColumns:
    if invalid:
        seqs = seqs[~seqs.isin(invalid)]
    seqs.dropna(inplace=True)
    #Over Top N
    uniques, counts, min_len, max_len, lengths = biological_data_top_entries(seqs, top_n)
    nucleotide_count = [dict(Counter(s)) for s in uniques]
    k_mers = _kmer_check(k, top, uniques)

    #Over all sequences
    all_overview = pd.DataFrame(seqs, columns=['sequence'])
    all_overview['sequence'] = seqs.str.upper()
    all_overview["lengths"] = all_overview['sequence'].str.len()

    A = all_overview['sequence'].str.count('A')
    T = all_overview['sequence'].str.count('T')
    G = all_overview['sequence'].str.count('G')
    C = all_overview['sequence'].str.count('C')
    all_overview['GC_skew'] = (G - C) / (G + C)
    all_overview['AT_skew'] = (A - T) / (A + T)

    all_overview["GC_count"] = all_overview['sequence'].str.count('[GC]')
    all_overview["gc_content"] = np.round(np.where(all_overview["lengths"] > 0, all_overview["GC_count"] / all_overview["lengths"] * 100, 0.0), 2)

    all_overview["N"] = all_overview['sequence'].str.count('N')
    all_overview["ambiguous_base_ratio"] = np.round(np.where(all_overview["lengths"] > 0, all_overview["N"] / all_overview["lengths"] * 100, 0.0), 2)

    all_overview["codon_complete"] = all_overview["lengths"] % 3
    all_overview["codon_pct"] = np.where(all_overview["lengths"] > 0, (all_overview["lengths"] - all_overview["codon_complete"]) / all_overview["lengths"] * 100,0.0)
    codon_completeness = SequenceMetricSummary(
        min=round(float(np.nanmin(all_overview["codon_pct"] )), 2),
        max=round(float(np.nanmax(all_overview["codon_pct"] )), 2),
        mean=round(float(np.nanmean(all_overview["codon_pct"] )), 2),
    )

    gc_content = SequenceMetricSummary(
        min=round(float(all_overview["gc_content"].min()), 2),
        max=round(float(all_overview["gc_content"].max()), 2),
        mean=round(float(all_overview["gc_content"].mean()), 2),
    )

    ambiguous_base_ratio = SequenceMetricSummary(
        min=round(float(all_overview["ambiguous_base_ratio"].min()), 2),
        max=round(float(all_overview["ambiguous_base_ratio"].max()), 2),
        mean=round(float(all_overview["ambiguous_base_ratio"].mean()), 2),
    )

    length_stats = SequenceMetricSummary(
        min=round(float(all_overview["lengths"].min()), 2),
        max=round(float(all_overview["lengths"].max()), 2),
        mean=round(float(all_overview["lengths"].mean()), 2),
    )

    gc_skew = SequenceMetricSummary(
        min=round(float(all_overview["GC_skew"].min()), 4),
        max=round(float(all_overview["GC_skew"].max()), 4),
        mean=round(float(all_overview["GC_skew"].mean()), 4),
    )

    at_skew = SequenceMetricSummary(
        min=round(float(all_overview["AT_skew"].min()), 4),
        max=round(float(all_overview["AT_skew"].max()), 4),
        mean=round(float(all_overview["AT_skew"].mean()), 4),
    )

    cpg_oe, tpa_oe = _dinucleotide_oe(all_overview)

    all_overview["entropy"] = all_overview["sequence"].apply(_normalized_shanon_entropy)

    low_complexity = SequenceMetricSummary(
        min=round(float(all_overview["entropy"].min()), 2),
        max=round(float(all_overview["entropy"].max()), 2),
        mean=round(float(all_overview["entropy"].mean()), 2),
    )

    reverse_complement_ratio, reverse_complement_list = _reverse_complement_duplicates(all_overview['sequence'])
    at_gc_plot = at_gc_skewness(all_overview)
    gc_dist_plot = gc_distribution(all_overview)

    affected_sequences = (all_overview["N"] > 0).sum()
    ambiguous_dist_plot = None
    if affected_sequences > 0:
        ambiguous_dist_plot = ambiguous_distribution(all_overview)

    length_dist_plot = None
    if all_overview["lengths"].nunique() > 1:
        length_dist_plot = length_distribution(all_overview)

    length_outliers = detect_outliers(all_overview["lengths"].to_numpy(dtype=np.double))


    if min_len == max_len:
        plot = make_logo(uniques, 'color_classic', seq_type="dna")
    else:
        flat_kmers = chain.from_iterable(k_mers)
        df_kmers = pd.DataFrame(flat_kmers, columns=['kmer', 'count'])
        aggregated = df_kmers.groupby('kmer', as_index=False)['count'].sum()
        plot = plot_overview(aggregated['kmer'].tolist(), aggregated['count'].tolist())

    return DNARNAColumns(
        sequence=uniques.tolist(),
        count=counts.tolist(),
        length=lengths.tolist(),
        gc_content=gc_content,
        ambiguous_base_ratio=ambiguous_base_ratio,
        length_stats=length_stats,
        length_outliers=length_outliers,
        codon_completeness=codon_completeness,
        gc_skew=gc_skew,
        at_skew=at_skew,
        cpg_observed_expected=cpg_oe,
        tpa_observed_expected=tpa_oe,
        low_complexity=low_complexity,
        reverse_complement_ratio=reverse_complement_ratio,
        reverse_complement_list=reverse_complement_list,
        nucleotide_count=nucleotide_count,
        k_mers=k_mers,
        plot=plot,
        gc_distribution=gc_dist_plot,
        length_distribution=length_dist_plot,
        ambiguous_distribution=ambiguous_dist_plot,
        at_gc_skewness=at_gc_plot,
    )

def _kmer_check(k: int, top: int, uniques: np.ndarray) -> list:
    check_length = any(len(i) <= k for i in uniques)
    if not check_length:
        k_mers = [top_mere(seq, n=k, top=top) for seq in uniques]
    else:
        print("Warning: Sequence length is smaller than choosen k-Mer Size. Setting k-Mer Size to 3.")
        k_mers = [top_mere(seq, n=3, top=top) for seq in uniques]
    return k_mers

def _reverse_complement_duplicates(all_seqs: pd.Series) -> Tuple[float,set]:
    complement = str.maketrans('ACGT', 'TGCA')
    seq_set = set(all_seqs)
    canonical = {min(seq, seq.translate(complement)[::-1]) for seq in seq_set}
    redundant_seqs = seq_set - canonical
    redundant = len(seq_set) - len(canonical)
    return (round(redundant / len(seq_set) * 100, 2) if len(seq_set) > 0 else 0.0), redundant_seqs

def _normalized_shanon_entropy(seq: str) -> np.float64:
    n = len(seq)
    if n == 0:
        return np.float64(0.0)
    counts = np.fromiter(Counter(seq).values(), dtype=np.float64)
    max_entropy = np.log2(min(n, len(counts)))  # ToDo or 4?
    if max_entropy == 0:
        return np.float64(0.0)
    p = counts / n
    h = -np.sum(p * np.log2(p))
    return np.float64(round(float(h / max_entropy * 100), 2))

def _dinucleotide_oe(df: pd.DataFrame) -> Tuple[SequenceMetricSummary, SequenceMetricSummary]:
    seqs = df["sequence"]
    lengths = df["lengths"]

    a_freq = seqs.str.count('A') / lengths
    t_freq = seqs.str.count('T') / lengths
    g_freq = seqs.str.count('G') / lengths
    c_freq = seqs.str.count('C') / lengths

    tpa_obs = seqs.str.count('TA') / (lengths - 1)
    tpa_exp = t_freq  * a_freq
    tpa_oe = np.where(tpa_exp > 0 , tpa_obs/tpa_exp, 0.0)

    cpg_obs = seqs.str.count('CG') / (lengths - 1)
    cpg_exp = g_freq * c_freq
    cpg_oe = np.where(cpg_exp > 0 , cpg_obs/cpg_exp, 0.0)

    #ToDo check if wie only use values above 0.0
    return (
        SequenceMetricSummary(
            min=round(float(np.nanmin(cpg_oe)), 4),
            max=round(float(np.nanmax(cpg_oe)), 4),
            mean=round(float(np.nanmean(cpg_oe)), 4),
        ),
        SequenceMetricSummary(
            min=round(float(np.nanmin(tpa_oe)), 4),
            max=round(float(np.nanmax(tpa_oe)), 4),
            mean=round(float(np.nanmean(tpa_oe)), 4),
        ),
    )

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

#ToDo Add Desclaimer; Removing invalid from list
def protein_columns(seqs: pd.Series, k: int = 3, top_n: int = 20, top: int = 5, invalid: list=None) -> PROTEINColumns:
    # Top-N
    if invalid:
        seqs = seqs[~seqs.isin(invalid)]
    uniques, counts, min_len, max_len, lengths = biological_data_top_entries(seqs, top_n)
    aa_composition = [dict(Counter(seq)) for seq in uniques]
    descriptors = [protein_descriptors(seq) for seq in uniques]
    k_mers = _kmer_check(k, top, uniques)

    # Column-wide
    all_overview = pd.DataFrame(seqs, columns=['sequence'])
    all_overview['sequence'] = seqs.str.upper()
    all_overview['lengths'] = all_overview['sequence'].str.len()

    all_overview['ambiguous'] = all_overview['sequence'].str.count('[XJU]')
    all_overview['ambiguous_ratio'] = np.round(np.where(all_overview['lengths'] > 0, all_overview['ambiguous'] / all_overview['lengths'] * 100, 0.0), 2)

    all_overview['stop_codon'] = all_overview['sequence'].str.count(r'\*')
    all_overview['entropy'] = all_overview['sequence'].apply(_normalized_shanon_entropy)

    ambiguous_residue_ratio = SequenceMetricSummary(
        min=round(float(all_overview['ambiguous_ratio'].min()), 2),
        max=round(float(all_overview['ambiguous_ratio'].max()), 2),
        mean=round(float(all_overview['ambiguous_ratio'].mean()), 2),
    )

    length_stats = SequenceMetricSummary(
        min=round(float(all_overview['lengths'].min()), 2),
        max=round(float(all_overview['lengths'].max()), 2),
        mean=round(float(all_overview['lengths'].mean()), 2),
    )

    length_outliers = detect_outliers(all_overview['lengths'].to_numpy(dtype=np.double))

    stop_codon_ratio = round((all_overview['stop_codon'] > 0).sum() / len(all_overview) * 100, 2)

    low_complexity = SequenceMetricSummary(
        min=round(float(all_overview['entropy'].min()), 2),
        max=round(float(all_overview['entropy'].max()), 2),
        mean=round(float(all_overview['entropy'].mean()), 2),
    )

    all_overview['gravy'] = all_overview['sequence'].apply(_gravy)
    all_overview['cysteine'] = all_overview['sequence'].str.count('C')
    all_overview['disorder'] = all_overview['sequence'].str.count(f"[{''.join(DISORDER_AA)}]")
    all_overview['disorder_ratio'] = np.round(
        np.where(all_overview['lengths'] > 0, all_overview['disorder'] / all_overview['lengths'] * 100, 0.0), 2
    )

    gravy = SequenceMetricSummary(
        min=round(float(all_overview['gravy'].min()), 4),
        max=round(float(all_overview['gravy'].max()), 4),
        mean=round(float(all_overview['gravy'].mean()), 4),
    )

    cysteine_count = SequenceMetricSummary(
        min=round(float(all_overview['cysteine'].min()), 2),
        max=round(float(all_overview['cysteine'].max()), 2),
        mean=round(float(all_overview['cysteine'].mean()), 2),
    )

    disorder_propensity = SequenceMetricSummary(
        min=round(float(all_overview['disorder_ratio'].min()), 2),
        max=round(float(all_overview['disorder_ratio'].max()), 2),
        mean=round(float(all_overview['disorder_ratio'].mean()), 2),
    )

    aa_group_dist = _aa_group_distribution(all_overview['sequence'])
    aa_group_plot = aa_group_distribution(aa_group_dist)

    length_dist_plot = None
    if all_overview['lengths'].nunique() > 1:
        length_dist_plot = length_distribution(all_overview, unit="residues")

    affected = (all_overview['ambiguous'] > 0).sum()
    ambiguous_dist_plot = None
    if affected > 0:
        ambiguous_dist_plot = ambiguous_distribution(all_overview, col='ambiguous', label='X/J/U')

    plot = None
    if min_len == max_len:
        plot = make_logo(uniques, "chemistry", seq_type="protein")
    if plot is None:
        flat_kmers = chain.from_iterable(k_mers)
        df_kmers = pd.DataFrame(flat_kmers, columns=['kmer', 'count'])
        aggregated = df_kmers.groupby('kmer', as_index=False)['count'].sum()
        plot = plot_overview(aggregated['kmer'].tolist(), aggregated['count'].tolist())

    return PROTEINColumns(
        sequence=uniques.tolist(),
        count=counts.tolist(),
        length=lengths.tolist(),
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
        ambiguous_residue_ratio=ambiguous_residue_ratio,
        length_stats=length_stats,
        length_outliers=length_outliers,
        stop_codon_ratio=stop_codon_ratio,
        low_complexity=low_complexity,
        k_mers=k_mers,
        plot=plot,
        length_distribution=length_dist_plot,
        ambiguous_distribution=ambiguous_dist_plot,
        gravy=gravy,
        cysteine_count=cysteine_count,
        disorder_propensity=disorder_propensity,
        aa_group_distribution=aa_group_dist,
        aa_group_plot=aa_group_plot
    )

AA_GROUPS = {
    "Unpolar":  set("GAVLIMP"),
    "Aromatic": set("FWY"),
    "Polar":    set("STCNQ"),
    "Positive": set("KRH"),
    "Negative": set("DE"),
}

DISORDER_AA = set("PESQK")
KYTE_DOOLITTLE = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2,
                  'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,}

def _gravy(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    try:
        return ProteinAnalysis(seq).gravy()
    except (KeyError, ValueError):
        # Fallback for sequences with ambiguous residues (X, J, U, *)
        return sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / len(seq)

def _aa_group_distribution(all_seqs: pd.Series) -> dict:
    concat = all_seqs.str.cat()
    total = len(concat)
    if total == 0:
        return {g: 0.0 for g in AA_GROUPS}
    counts = Counter(concat)
    return {
        group: round(sum(counts.get(aa, 0) for aa in members) / total, 4)
        for group, members in AA_GROUPS.items()
    }

def make_logo(seqs, color, seq_type):
    if seq_type == "protein":
        m = motifs.create(seqs, alphabet="ACDEFGHIKLMNPQRSTVWY")
    else:
        m = motifs.create(seqs, alphabet="ACGT")
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        m.weblogo(tmp_path, format="svg", sequence_type=seq_type, color_scheme=color, logo_font="Calibri", logo_margin=3,
                  fontsize=12, scale_width=True)

        with open(tmp_path, 'r', encoding='utf-8') as svg_file:
            raw = svg_file.read()

        # WebLogo 3 wraps the SVG in a full HTML document — extract only the SVG fragment
        # to avoid injecting Bootstrap 3, jQuery 1.x and other legacy dependencies into the report
        svg_match = re.search(r'(<svg[\s\S]*?</svg>)', raw, re.IGNORECASE)
        if svg_match:
            svg_content = svg_match.group(1)
            svg_content = svg_content.replace(
                '<svg ',
                '<svg style="width: 100%; height: 250px; max-width: 800px;" '
            )
            return svg_content

        return None

    except (HTTPError, URLError, OSError) as e:
        print(f"WebLogo Connection Error: {e}")
        return None

    finally:
        if Path(tmp_path).is_file():
            Path(tmp_path).unlink(missing_ok=True)


def plot_overview(kmer, count):
    fig = px.bar(x=kmer, y=count, color_discrete_sequence=['#0F65A0'])
    fig.update_layout(bargap=0.2, plot_bgcolor='white', xaxis_title='K-mers',
                      yaxis_title='Count')
    apply_standard_axes(fig, tick_angle=-45)
    return fig.to_html(full_html=False, include_plotlyjs=False)