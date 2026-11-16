# Data directory

The Legatum data files are not included in this repository; they are published by the Legatum Institute Foundation under its terms of use.

Place both files here, or set the environment variables:

| File | SHA-256 | Variable |
| :-- | :-- | :-- |
| `Dataset_Legatum_Prosperity_Index_2023.xlsx` | `8c789bd5ab881c12005f83e9b0c55ae81a68548dd18d3acb9fff9bec0eef631c` | `LPI_DATA_2023` |
| `Legatum_Prosperity_Index_2026_Data_Sheet.xlsx` | `54621ff5d00d0cf4afe6bdbcc47008acff2086cd87cf55c3c8777f819eed1718` | `LPI_DATA_2026` |

The 2023 edition has been withdrawn from the publisher's website. The analysis reads its sheets `Prosperity Index` (keyed on `area_code`, columns `score_2023`, `rank_2023`) and `Pillars x 12` (long format: `area_code`, `pillar_name`, `score_2023`). From the 2026 file it reads the `ranks*` sheets and `scores_uni_wghts_uni_p`.

The cross-index comparison of Section 8.3 (`src/cross_index.py`) also reads four public files, downloaded on 15 November 2026 and not redistributed here:

| File | Publisher and URL | SHA-256 | Variable |
| :-- | :-- | :-- | :-- |
| `HDR25_Composite_indices_complete_time_series.csv` | UNDP, https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv | `61ed82e5b66c88dfca8ff9fac775c63981ecab6a254862af97acacc41c143117` | `HDI_DATA` |
| `wipo-pub-2000-2024-tech1.xlsx` | WIPO, https://www.wipo.int/edocs/pubdocs/en/wipo-pub-2000-2024-tech1.xlsx | `f7ab52221b82d15645eacdaf6f0ca0c1ce7b3686f814eb15f9bed8c7d9d88126` | `GII_DATA` |
| `WGI_2026_Governance_Estimates_and_Scores_1996-2025.xlsx` (published as `WGI 2026 Governance Estimates and Scores (1996-2025).xlsx`) | World Bank, https://datacatalogfiles.worldbank.org/ddh-published/0038026/DR0095947/WGI%202026%20Governance%20Estimates%20and%20Scores%20(1996-2025).xlsx | `8e4754bda8b2be0b36d172c18a4caa634cb36fddd54145d16ab7316d22b3542c` | `WGI_DATA` |
| `database_2026.xlsx` | SDSN, https://dashboards.sdgindex.org/static/downloads/database_2026.xlsx | `ac34ef41945cf3cb69d420841e66eef6bf4d43dc5a76186e0c5a2ed658c7b892` | `SDG_DATA` |

These files remain under their publishers' terms; the SDSN database is licensed CC BY-NC-ND 4.0 (cite: Sachs, J. D., Lafortune, G., Fuller, G., & Iablonovski, G. (2026), *Sustainable Development Report 2026*, https://doi.org/10.82163/225), so none of them is redistributed here. The committed outputs contain only derived statistics.

Then `python src/cross_index.py` writes `outputs/cross_index.csv` and `outputs/cross_index.json`, and `python src/figures.py` adds Figure 9.

To confirm a copy is the same data, run `pytest -q tests/`: it checks the SHA-256 digest and that the equal-weighted mean of the twelve pillar scores reproduces every published 2023 score and rank.
