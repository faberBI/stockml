import numpy as np

from stockml.features import FEATURES, TARGET, build_features
from stockml.train import date_splits


def test_no_nan_and_target_is_next_return(prices):
    df = build_features(prices)
    assert not df[FEATURES + [TARGET]].isna().any().any()
    g = df[df.ticker == "AAA"].set_index("date")
    p = prices[prices.ticker == "AAA"].set_index("date")["close"]
    d = g.index[10]
    nxt = p.index[p.index.get_loc(d) + 1]
    assert np.isclose(g.loc[d, TARGET], np.log(p[nxt] / p[d]))


def test_walk_forward_has_no_overlap(prices):
    df = build_features(prices)
    for tr, te in date_splits(df["date"], 4):
        assert df["date"].iloc[tr].max() < df["date"].iloc[te].min()
