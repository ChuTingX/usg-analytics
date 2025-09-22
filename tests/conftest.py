import os

import numpy as np
import pandas as pd


def make_synth_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2020-01-01", periods=n)
    base = rng.normal(0, 1, size=n).cumsum() + 50
    df = pd.DataFrame(
        {
            "Date": idx,
            "Adj Close": base + rng.normal(0, 0.5, size=n),
            "SP_close": base * 0.1 + rng.normal(0, 1, size=n),
            "DJ_close": base * 0.2 + rng.normal(0, 1, size=n),
            "USDI_Price": 100 + rng.normal(0, 1, size=n),
            "EU_Price": 80 + rng.normal(0, 1, size=n),
            "GDX_Close": base * 0.05 + rng.normal(0, 1, size=n),
            "SF_Price": 10 + rng.normal(0, 1, size=n),
            "PLT_Price": 20 + rng.normal(0, 1, size=n),
            "PLD_Price": 30 + rng.normal(0, 1, size=n),
            "RHO_PRICE": 40 + rng.normal(0, 1, size=n),
            "USO_Close": base * 0.15 + rng.normal(0, 1, size=n),
            "OF_Price": 5 + rng.normal(0, 1, size=n),
            "OS_Price": 6 + rng.normal(0, 1, size=n),
            "Adj Close_Trend": rng.normal(0, 1, size=n),
        }
    )
    return df


def write_synth_csv(tmpdir):
    df = make_synth_df()
    csv_path = os.path.join(tmpdir, "sample.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
