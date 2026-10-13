"""Registry dei modelli candidati (~50): famiglie diverse x iperparametri diversi.

Per aggiungere un modello basta aggiungere una voce al dizionario in `get_models()`.
"""
from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_STATE

RS = RANDOM_STATE


def _pipe(est, poly: bool = False) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if poly:  # interazioni di 2° ordine tra le feature, poi ri-scalate
        steps += [("poly", PolynomialFeatures(2, include_bias=False)),
                  ("scaler2", StandardScaler())]
    return Pipeline([*steps, ("model", est)])


def _scaled_y(est):
    """Standardizza anche il target: i rendimenti (~1e-2) rendono instabili
    i modelli con penalità/learning rate in scala assoluta (Lasso, SVR, SGD, MLP)."""
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def _linear() -> dict:
    m = {"ols": LinearRegression()}
    for a in (0.1, 1, 10, 100, 1000):
        m[f"ridge_a{a:g}"] = Ridge(alpha=a)
    for a in (0.001, 0.01, 0.05):
        m[f"lasso_a{a:g}"] = _scaled_y(Lasso(alpha=a, max_iter=5000))
    for a in (0.01, 0.1):
        for r in (0.2, 0.8):
            m[f"enet_a{a:g}_l1{r:g}"] = _scaled_y(ElasticNet(alpha=a, l1_ratio=r, max_iter=5000))
    m["huber"] = _scaled_y(HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000))
    m["bayesian_ridge"] = BayesianRidge()
    m["ard"] = ARDRegression()
    m["sgd_huber"] = _scaled_y(SGDRegressor(loss="huber", alpha=1e-3, random_state=RS))
    m["pls_2"] = PLSRegression(n_components=2)
    m["pls_5"] = PLSRegression(n_components=5)
    m["ridge_poly2"] = ("poly", Ridge(alpha=100))
    m["lasso_poly2"] = ("poly", _scaled_y(Lasso(alpha=0.01, max_iter=5000)))
    return m


def _neighbors_svm() -> dict:
    m = {}
    for k in (25, 50, 100, 200):
        m[f"knn_k{k}"] = KNeighborsRegressor(n_neighbors=k, weights="distance")
    m["knn_k100_uniform"] = KNeighborsRegressor(n_neighbors=100, weights="uniform")
    m["linear_svr"] = _scaled_y(LinearSVR(C=0.1, epsilon=0.1, max_iter=5000, random_state=RS))
    for c in (0.1, 1.0):
        m[f"svr_rbf_c{c:g}"] = _scaled_y(SVR(C=c, epsilon=0.1, gamma="scale"))
    return m


def _trees() -> dict:
    m = {
        "tree_d3": DecisionTreeRegressor(max_depth=3, min_samples_leaf=100, random_state=RS),
        "tree_d6": DecisionTreeRegressor(max_depth=6, min_samples_leaf=100, random_state=RS),
    }
    for d in (4, 8):
        m[f"rf_d{d}"] = RandomForestRegressor(
            n_estimators=200, max_depth=d, min_samples_leaf=50, max_features=0.5,
            n_jobs=-1, random_state=RS)
        m[f"extratrees_d{d}"] = ExtraTreesRegressor(
            n_estimators=200, max_depth=d, min_samples_leaf=50, max_features=0.5,
            n_jobs=-1, random_state=RS)
    m["bagging_tree"] = BaggingRegressor(
        DecisionTreeRegressor(max_depth=6, min_samples_leaf=50),
        n_estimators=50, n_jobs=-1, random_state=RS)
    m["adaboost"] = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=3), n_estimators=100, learning_rate=0.05,
        random_state=RS)
    m["gbr_huber"] = GradientBoostingRegressor(
        loss="huber", n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
        random_state=RS)
    for d, lr in ((2, 0.05), (3, 0.03), (5, 0.02)):
        m[f"hgb_d{d}_lr{lr:g}"] = HistGradientBoostingRegressor(
            max_depth=d, learning_rate=lr, max_iter=300, l2_regularization=1.0,
            min_samples_leaf=50, random_state=RS)
    return m


def _boosting_libs() -> dict:
    """LightGBM/XGBoost: inclusi solo se installati (sono nelle dipendenze del pacchetto)."""
    m = {}
    try:
        from lightgbm import LGBMRegressor
        for leaves in (7, 15, 31):
            m[f"lgbm_l{leaves}"] = LGBMRegressor(
                n_estimators=300, num_leaves=leaves, learning_rate=0.03, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                reg_lambda=1.0, random_state=RS, verbose=-1)
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        for d in (2, 3, 4):
            m[f"xgb_d{d}"] = XGBRegressor(
                n_estimators=300, max_depth=d, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=RS, n_jobs=-1)
    except ImportError:
        pass
        try:
        from catboost import CatBoostRegressor
        for d in (4, 6):
            m[f"catboost_d{d}"] = CatBoostRegressor(
                iterations=300, depth=d, learning_rate=0.03, l2_leaf_reg=3.0,
                random_seed=RS, verbose=0, thread_count=-1,
                allow_writing_files=False)  # evita la cartella catboost_info/ nel repo
        m["catboost_d4_rmsewithunc"] = CatBoostRegressor(
            iterations=300, depth=4, learning_rate=0.03, loss_function="Huber:delta=0.02",
            random_seed=RS, verbose=0, thread_count=-1, allow_writing_files=False)
    except ImportError:
        pass
    return m


def _neural() -> dict:
    m = {}
    for name, layers, alpha in (("mlp_small", (16,), 1.0), ("mlp_medium", (32, 16), 1.0),
                                ("mlp_large", (64, 32), 3.0)):
        m[name] = _scaled_y(MLPRegressor(
            hidden_layer_sizes=layers, alpha=alpha, early_stopping=True,
            max_iter=500, random_state=RS))
    return m


def get_models() -> dict[str, Pipeline]:
    raw = {**_linear(), **_neighbors_svm(), **_trees(), **_boosting_libs(), **_neural()}
    out = {}
    for name, est in raw.items():
        if isinstance(est, tuple):  # ("poly", estimator)
            out[name] = _pipe(est[1], poly=True)
        else:
            out[name] = _pipe(est)
    return out
