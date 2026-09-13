# Data directory

The Legatum data files are not included in this repository; they are published by the Legatum Institute Foundation under its terms of use.

Place both files here, or set the environment variables:

| File | SHA-256 | Variable |
| :-- | :-- | :-- |
| `Dataset_Legatum_Prosperity_Index_2023.xlsx` | `8c789bd5ab881c12005f83e9b0c55ae81a68548dd18d3acb9fff9bec0eef631c` | `LPI_DATA_2023` |
| `Legatum_Prosperity_Index_2026_Data_Sheet.xlsx` | `54621ff5d00d0cf4afe6bdbcc47008acff2086cd87cf55c3c8777f819eed1718` | `LPI_DATA_2026` |

The 2023 edition has been withdrawn from the publisher's website. The analysis reads its sheets `Prosperity Index` (keyed on `area_code`, columns `score_2023`, `rank_2023`) and `Pillars x 12` (long format: `area_code`, `pillar_name`, `score_2023`). From the 2026 file it reads the `ranks*` sheets and `scores_uni_wghts_uni_p`.

To confirm a copy is the same data, run `pytest -q tests/`: it checks the SHA-256 digest and that the equal-weighted mean of the twelve pillar scores reproduces every published 2023 score and rank.
