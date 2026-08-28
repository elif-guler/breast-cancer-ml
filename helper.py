import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def setup_folders():
    """Create the project's output folders if they don't already exist."""
    for folder in ["data", "plots", "models", "results"]:
        os.makedirs(folder, exist_ok=True)
        print(f"Ready: {folder}/")


def save_dataset(df, name):
    """Save a DataFrame (or anything array-like) to data/<name>.csv."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    path = os.path.join("data", f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved dataset -> {path}")


def save_plot(fig, name, dpi=300):
    """Save a matplotlib figure to plots/<name>.png."""
    path = os.path.join("plots", f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot -> {path}")
    plt.close(fig)


def save_model(model, name):
    """Save a fitted model to models/<name>.joblib."""
    path = os.path.join("models", f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Saved model -> {path}")


def save_results(results_dict, name="results_models"):
    """Save a {model_name: {metric: value}} dict to results/<name>.csv."""
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    path = os.path.join("results", f"{name}.csv")
    df.to_csv(path, index_label="Model")
    print(f"Saved results -> {path}")
