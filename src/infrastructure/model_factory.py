from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    HuberRegressor, PassiveAggressiveRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    BaggingRegressor, ExtraTreesRegressor, 
    AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib

def get_linear_models():
    """Returns basic linear models."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Huber": HuberRegressor(max_iter=1000),
        "PassiveAggressive": PassiveAggressiveRegressor(random_state=42)
    }

def get_tree_models():
    """Returns tree-based models."""
    if config.FAST_MODE:
        return {
            "DecisionTree": DecisionTreeRegressor()
        }
    return {
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42)
    }

def get_other_models():
    """Returns other miscellaneous models."""
    if config.FAST_MODE:
        return {
            "KNN": KNeighborsRegressor()
        }
    return {
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Bagging": BaggingRegressor(random_state=42),
        "MLPNN": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

def get_models():
    """
    Returns a unified dictionary of all initialized models.
    """
    models = {}
    models.update(get_linear_models())
    models.update(get_tree_models())
    models.update(get_other_models())
    return models

def train_and_evaluate(model, x_train, x_test, y_train, y_test):
    """
    Trains a model and returns its performance metrics.
    """
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return model, {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}

import os
from src.shared import config

def save_model(model, name):
    """Saves a model to the models directory."""
    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)
    path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    return path

def run_all_models(x_train, x_test, y_train, y_test):
    """Trains, evaluates and saves multiple models."""
    results = {}
    models = get_models()
    for name, model in models.items():
        print(f"Training {name}...")
        trained_model, metrics = train_and_evaluate(
            model, x_train, x_test, y_train, y_test
        )
        save_model(trained_model, name)
        print(f"Finished {name}.")
        results[name] = {"model": trained_model, "metrics": metrics}
    return results
