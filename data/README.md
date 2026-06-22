# Data directory

The Legatum Prosperity Index dataset is **not** included in this repository
because it is distributed under the Prosperity Institute's terms of use.

To run the analysis, download the file yourself and place it here:

1. Go to https://index.prosperity.com (Resources / FAQ, "Can I see the actual data used?").
2. Download `Dataset_Legatum_Prosperity_Index_2023.xlsx`.
3. Save it in this `data/` folder, or set an environment variable:

       export LPI_DATA=/path/to/Dataset_Legatum_Prosperity_Index_2023.xlsx

The code looks for `data/Dataset_Legatum_Prosperity_Index_2023.xlsx` by default,
or the path in the `LPI_DATA` environment variable if set.
