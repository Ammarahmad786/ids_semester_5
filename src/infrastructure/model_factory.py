from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    BaggingRegressor, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def get_linear_models():
    """Returns basic linear models."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet()
    }

def get_tree_models():
    """Returns tree-based models."""
    return {
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

def get_other_models():
    """Returns other miscellaneous models."""
    return {
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Bagging": BaggingRegressor(random_state=42)
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
    r2 = r2_score(y_test, predictions)
    return model, {"MSE": mse, "R2": r2}

import os
from src.shared.config import MODELS_DIR

def save_model(model, name):
    """Saves a model to the models directory."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
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
