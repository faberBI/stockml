"""I 5 modelli candidati. Aggiungerne uno = aggiungere una riga al registry."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def get_models() -> dict[str, Pipeline]:
    def pipe(est):
        return Pipeline([("scaler", StandardScaler()), ("model", est)])

    return {
        "ridge": pipe(Ridge(alpha=10.0)),
        "random_forest": pipe(RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            n_jobs=-1, random_state=RANDOM_STATE)),
        "hist_gradient_boosting": pipe(HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
            random_state=RANDOM_STATE)),
        "knn": pipe(KNeighborsRegressor(n_neighbors=100, weights="distance")),
        # target standardizzato: i rendimenti (~1e-2) destabilizzano l'ottimizzatore della rete
        "mlp": pipe(TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, early_stopping=True,
                                   max_iter=500, random_state=RANDOM_STATE),
            transformer=StandardScaler())),
    }
