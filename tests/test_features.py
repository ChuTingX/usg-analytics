from code_usg.features import add_target_rollings, drop_trend_columns, make_lagged, select_variables

from .conftest import make_synth_df


def test_feature_pipeline_basic():
    df = make_synth_df()
    df = df.set_index("Date")
    df2 = drop_trend_columns(df, "_Trend")
    df3 = select_variables(df2, ["Adj Close", "SP_close", "DJ_close"])
    df4 = add_target_rollings(df3, "Adj Close", [7, 30]).dropna()
    lagged = make_lagged(df4, list(df4.columns), "Adj Close", lag=1)
    assert "Adj Close_prev" in lagged.columns
    assert not lagged.isna().any().any()
