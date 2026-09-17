from functools import lru_cache
import pandas as pd

SUNBURST_RANKS = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
SUNBURST_PATH = ["superdomain"] + SUNBURST_RANKS
RANK_ALIASES = {"superkingdom": "domain", "realm": "domain"}

class TaxonomyLineageResolver:
    VIRUS_ROOT_TAXID = 10239  # NCBI taxid for virus

    def __init__(self, raw_tax_df: pd.DataFrame):
        nodes = raw_tax_df[['tax_id', 'parent_tax_id', 'rank']].drop_duplicates('tax_id').set_index('tax_id')
        self.names = (raw_tax_df[raw_tax_df["name_class"] == "scientific name"].drop_duplicates("tax_id").set_index('tax_id')["name_txt"])
        self.parent_map = nodes["parent_tax_id"].to_dict()
        self.rank_map = nodes["rank"].to_dict()

    @lru_cache(maxsize=None)
    def resolve(self, tax_id) -> tuple[dict, bool]:
        rank = RANK_ALIASES.get(self.rank_map.get(tax_id), self.rank_map.get(tax_id))
        own = {rank: self.names.get(tax_id, str(tax_id))} if rank in SUNBURST_RANKS else {}
        is_this_virus = tax_id == self.VIRUS_ROOT_TAXID

        parent_tmp = self.parent_map.get(tax_id)
        if parent_tmp is None or parent_tmp == tax_id:
            return own, is_this_virus

        parent, parent_is_virus = self.resolve(parent_tmp)
        return {**parent, **own}, parent_is_virus or is_this_virus

    def build_lineage(self, tax_ids: set) -> pd.DataFrame:
        columns = ["tax_id", "scientific_name", "superdomain"] + SUNBURST_RANKS
        if not tax_ids:
            return pd.DataFrame(columns=columns)

        rows = list()
        for tax_id in tax_ids:
            chain, is_virus = self.resolve(tax_id)
            filled = pd.Series([chain.get(r) for r in SUNBURST_RANKS]).bfill().ffill().tolist()

            row = {
                "tax_id": tax_id,
                "scientific_name": self.names.get(tax_id),
                "superdomain": "Viruses" if is_virus else filled[0],
            }
            row.update(dict(zip(SUNBURST_RANKS, filled)))
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)
