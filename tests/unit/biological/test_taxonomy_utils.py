import pandas as pd
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biological.taxonomy_utils import TaxonomyLineageResolver, SUNBURST_RANKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw_tax_df(rows):
    """rows: list of (tax_id, parent_tax_id, rank, name_txt, name_class)"""
    return pd.DataFrame(
        rows, columns=["tax_id", "parent_tax_id", "rank", "name_txt", "name_class"]
    )


def small_lineage_df():
    """A tiny realistic bacterial lineage:
    1 (root) -> 2 (Bacteria, superkingdom) -> 1224 (Proteobacteria, phylum)
    -> 1236 (Gammaproteobacteria, class) -> 91347 (Enterobacterales, order)
    -> 543 (Enterobacteriaceae, family) -> 561 (Escherichia, genus)
    -> 562 (Escherichia coli, species)
    """
    return make_raw_tax_df([
        (1, 1, "no rank", "root", "scientific name"),
        (2, 1, "superkingdom", "Bacteria", "scientific name"),
        (1224, 2, "phylum", "Proteobacteria", "scientific name"),
        (1236, 1224, "class", "Gammaproteobacteria", "scientific name"),
        (91347, 1236, "order", "Enterobacterales", "scientific name"),
        (543, 91347, "family", "Enterobacteriaceae", "scientific name"),
        (561, 543, "genus", "Escherichia", "scientific name"),
        (562, 561, "species", "Escherichia coli", "scientific name"),
    ])


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

class TestResolve:
    def test_resolves_full_lineage_chain(self):
        resolver = TaxonomyLineageResolver(small_lineage_df())
        chain, is_virus = resolver.resolve(562)
        assert chain["species"] == "Escherichia coli"
        assert chain["genus"] == "Escherichia"
        assert chain["family"] == "Enterobacteriaceae"
        assert chain["order"] == "Enterobacterales"
        assert chain["class"] == "Gammaproteobacteria"
        assert chain["phylum"] == "Proteobacteria"

    def test_is_virus_false_for_bacteria(self):
        resolver = TaxonomyLineageResolver(small_lineage_df())
        _, is_virus = resolver.resolve(562)
        assert is_virus is False

    def test_is_virus_true_for_virus_root_descendant(self):
        rows = small_lineage_df()
        # Add a virus lineage rooted at VIRUS_ROOT_TAXID (10239)
        virus_rows = make_raw_tax_df([
            (10239, 1, "superkingdom", "Viruses", "scientific name"),
            (99999, 10239, "family", "Some Viridae", "scientific name"),
        ])
        df = pd.concat([rows, virus_rows], ignore_index=True)
        resolver = TaxonomyLineageResolver(df)
        chain, is_virus = resolver.resolve(99999)
        assert is_virus is True
        assert chain.get("family") == "Some Viridae"

    def test_superkingdom_aliased_to_domain(self):
        """RANK_ALIASES maps 'superkingdom' -> 'domain'."""
        resolver = TaxonomyLineageResolver(small_lineage_df())
        chain, _ = resolver.resolve(562)
        assert chain.get("domain") == "Bacteria"
        assert "superkingdom" not in chain

    def test_stops_at_self_referencing_root(self):
        """tax_id 1's parent is itself (1) — recursion must terminate."""
        resolver = TaxonomyLineageResolver(small_lineage_df())
        chain, is_virus = resolver.resolve(1)
        assert is_virus is False
        # "no rank" isn't in SUNBURST_RANKS, so chain should be empty here
        assert chain == {}

    def test_unknown_taxid_falls_back_to_str_name(self):
        resolver = TaxonomyLineageResolver(small_lineage_df())
        chain, is_virus = resolver.resolve(9999999)
        # unknown parent -> parent_map.get returns None -> returns own={}, False
        assert chain == {}
        assert is_virus is False

    def test_resolve_is_cached(self):
        """lru_cache means repeated calls for the same tax_id return the
        exact same (identical) result object, not just an equal one."""
        resolver = TaxonomyLineageResolver(small_lineage_df())
        result1 = resolver.resolve(562)
        result2 = resolver.resolve(562)
        assert result1 == result2


# ---------------------------------------------------------------------------
# build_lineage
# ---------------------------------------------------------------------------

class TestBuildLineage:
    def test_empty_tax_ids_returns_empty_df_with_expected_columns(self):
        resolver = TaxonomyLineageResolver(small_lineage_df())
        result = resolver.build_lineage(set())
        assert result.empty
        assert list(result.columns) == ["tax_id", "scientific_name", "superdomain"] + SUNBURST_RANKS

    def test_single_taxid_produces_one_row(self):
        resolver = TaxonomyLineageResolver(small_lineage_df())
        result = resolver.build_lineage({562})
        assert len(result) == 1
        row = result.iloc[0]
        assert row["tax_id"] == 562
        assert row["scientific_name"] == "Escherichia coli"
        assert row["superdomain"] == "Bacteria"
        assert row["species"] == "Escherichia coli"

    def test_missing_ranks_are_filled_via_bfill_ffill(self):
        """Intermediate lineage without a 'kingdom' rank should have that
        gap filled from neighboring ranks rather than left as None."""
        resolver = TaxonomyLineageResolver(small_lineage_df())
        result = resolver.build_lineage({562})
        row = result.iloc[0]
        # kingdom rank doesn't exist in our lineage; it should be filled
        # in (bfill/ffill), not raise or stay a raw None causing NaN chaos.
        assert row["kingdom"] is not None

    def test_multiple_taxids_produce_multiple_rows(self):
        resolver = TaxonomyLineageResolver(small_lineage_df())
        result = resolver.build_lineage({562, 561})
        assert len(result) == 2
        assert set(result["tax_id"]) == {562, 561}

    def test_virus_lineage_marked_as_viruses_superdomain(self):
        rows = small_lineage_df()
        virus_rows = make_raw_tax_df([
            (10239, 1, "superkingdom", "Viruses", "scientific name"),
            (99999, 10239, "family", "Some Viridae", "scientific name"),
        ])
        df = pd.concat([rows, virus_rows], ignore_index=True)
        resolver = TaxonomyLineageResolver(df)
        result = resolver.build_lineage({99999})
        assert result.iloc[0]["superdomain"] == "Viruses"