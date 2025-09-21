import numpy as np
from code_usg.modeling import build_cv, baseline_mse_cv, random_forest_cv, all_subset_lr_cv, lasso_ridge_cv
from code_usg.features import make_lagged, add_target_rollings, select_variables, drop_trend_columns
from .conftest import make_synth_df

def test_modeling_cv_roundtrip():
    df = make_synth_df().set_index("Date")
    df = drop_trend_columns(df, "_Trend")
    df = select_variables(df, ["Adj Close","SP_close","DJ_close","USO_Close"])
    df = add_target_rollings(df, "Adj Close", [7, 30]).dropna()
    base_predictors = list(df.columns)
    lagged = make_lagged(df, base_predictors, "Adj Close", lag=1)
    y = lagged["Adj Close"].values
    predictor_cols = [c for c in lagged.columns if c.endswith("_prev")]
    X = lagged[predictor_cols].values
    Xb = lagged["Adj Close_prev"].values

    cv = build_cv("kfold", n_splits=3, shuffle=True, random_state=0)
    base_mses, base_mean = baseline_mse_cv(y, Xb, cv)
    rf_mses, rf_mean, rf_est = random_forest_cv(X, y, cv, {"n_estimators": 10, "random_state": 0})
    lr_mses, lr_mean, best_idx, best_feats = all_subset_lr_cv(X, y, cv, predictor_cols)
    lasso_mses, lasso_mean, alpha, est = lasso_ridge_cv("lasso", X, y, cv, {"alphas_logspace":[-4,1,5], "random_state":0})

    assert np.isfinite(base_mean)
    assert np.isfinite(rf_mean)
    assert len(best_feats) >= 1
    assert alpha > 0