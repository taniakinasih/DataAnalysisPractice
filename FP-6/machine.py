import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import parse_data as ps
import descriptive as ds


# Define the SET_E from previous classical.py analysis
SET_E = {
    "Winter": ["dew point", "pressure", "wind speed", "temperature", "rainfall duration"],
    "Spring": ["dew point", "temperature", "pressure", "wind speed"],
    "Summer": ["dew point", "pressure", "rainfall duration", "temperature", "wind speed"],
    "Autumn": ["dew point", "temperature", "wind speed", "rainfall duration"],
}

# Define Seasons and Kernels
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
KERNELS = ["linear", "poly", "rbf", "sigmoid"]


# Load Data
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
  
    df = ps.parse(filepath)
    df = ds.remove_outliers(df)
    return df


# 1st: SVR Kernel Comparison
def compare_svr_kernels(df: pd.DataFrame, features_dict: dict):
    
    fig, axes = plt.subplots(len(SEASONS), len(KERNELS), figsize=(20, 18))

    for i, season in enumerate(SEASONS):
        df_s = df[df["season"] == season].copy()
        X = df_s[features_dict[season]].values
        y = df_s["concentration"].values

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for j, kernel in enumerate(KERNELS):
            model = svm.SVR(kernel=kernel, C=100, epsilon=0.1)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            ax = axes[i, j]
            ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="white")
            ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],"r--",lw=1.5,)
            ax.set_title(f"{season} - {kernel}\nR²={r2:.3f} | MAE={mae:.2f}")

            if j == 0:
                ax.set_ylabel("Predicted PM2.5")
            if i == len(SEASONS) - 1:
                ax.set_xlabel("Actual PM2.5")

            ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("SVR Kernel Comparison Across Seasons", fontsize=22)
    plt.tight_layout()
    plt.show()


# 2nd: Hyperparameter Tuning (C and Epsilon Analysis)

def tune_svr_hyperparameters(
    df: pd.DataFrame,
    features_dict: dict,
    trials=((1, 1.0), (100, 0.1), (1000, 0.01)), #this is [C, Epsilon]
):
    
    for season in SEASONS:
        df_s = df[df["season"] == season].copy()
        X = df_s[features_dict[season]].values
        y = df_s["concentration"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        fig, axes = plt.subplots(1, len(trials), figsize=(6 * len(trials), 5))

        for i, (c_val, eps_val) in enumerate(trials):
            model = svm.SVR(kernel="rbf", C=c_val, epsilon=eps_val)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            mae = mean_absolute_error(y_test, y_pred)

            ax = axes[i]
            ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="white")
            ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],"r--",lw=2,)
            ax.set_title(f"{season}\nC={c_val}, ε={eps_val}\nMAE={mae:.2f}")
            ax.set_xlabel("Actual PM2.5")
            if i == 0:
                ax.set_ylabel("Predicted PM2.5")
            ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.show()

# 3rd: Gamma Analysis
def run_gamma_analysis(df, features_dict, gamma_list=['scale', 'auto'], c_val=1000, eps_val=0.1):

    fig, axes = plt.subplots(len(SEASONS), len(gamma_list), figsize=(15, 18))

    for i, season in enumerate(SEASONS):
        
        df_s = df[df["season"] == season].copy()
        features = features_dict[season]
        
        df_s = df_s.dropna(subset=features + ["concentration"])
        
        X = df_s[features].values
        y = df_s["concentration"].values
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        
        for j, g in enumerate(gamma_list):
            model = svm.SVR(kernel='rbf', C=c_val, epsilon=eps_val, gamma=g)
            model.fit(X_train_s, y_train)
            
            
            y_pred = model.predict(X_test_s)
            
            mae = mean_absolute_error(y_test, y_pred)
            
            ax = axes[i, j]
            ax.scatter(y_test, y_pred, alpha=0.4, color='teal', edgecolors='white')
            
            # Ideal line 1:1
            min_val, max_val = y_test.min(), y_test.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_title(f"{season}\nGamma: {g}\nMAE: {mae:.2f}")
            if j == 0: 
                ax.set_ylabel("Predicted Concentration")
            if i == len(SEASONS) - 1: 
                ax.set_xlabel("Actual Concentration")
            ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("SVR Gamma Analysis: Influence of Gamma on RBF Kernel Accuracy", fontsize=22, y=1.02)
    plt.tight_layout()
    plt.show()

# 4th: Optimized SVR with Lag + Log and Generate the Residual PLots

def analyze_optimized_svr(
    df: pd.DataFrame, seasons: list, features_dict: dict, c_val=1000, eps_val=0.01, sample_size=5000,) -> pd.DataFrame:
    df_work = df.copy()
    df_work["lag_1"] = df_work["concentration"].shift(1)
    df_final = df_work.dropna(subset=["lag_1"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    results = []

    for i, season in enumerate(seasons):
        df_s = df_final[df_final["season"] == season].copy()
        if len(df_s) > sample_size:
            df_s = df_s.sample(sample_size, random_state=42)

        features = features_dict[season] + ["lag_1"]
        X = df_s[features].values
        y = df_s["concentration"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        y_train_log = np.log1p(y_train)

        model = svm.SVR(kernel="rbf", C=c_val, epsilon=eps_val, cache_size=1500)
        model.fit(X_train_s, y_train_log)

        y_pred = np.expm1(model.predict(X_test_s))

        mae = mean_absolute_error(y_test, y_pred)
        results.append({"Season": season, "MAE": round(mae, 2)})

        ax = axes[i]
        ax.scatter(y_pred, y_test - y_pred, alpha=0.5, edgecolors="white")
        ax.axhline(0, linestyle="--", color="black")
        ax.set_title(f"{season}\nMAE={mae:.2f}")
        ax.set_xlabel("Predicted PM2.5")
        ax.set_ylabel("Residuals")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Optimized SVR: Lag + Log Transformation", fontsize=18)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results)


# 5th: Final SVR Evaluation
def analyze_final_svr_performance(
    df: pd.DataFrame,
    seasons: list,
    features_dict: dict,
    c_val=1000,
    eps_val=0.01,
    sample_size=5000,
) -> pd.DataFrame:
    """
    Final SVR evaluation with lag feature and performance metrics.
    """
    df_work = df.copy()
    df_work["lag_1"] = df_work["concentration"].shift(1)
    df_final = df_work.dropna(subset=["lag_1"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    summary = []

    for i, season in enumerate(seasons):
        df_s = df_final[df_final["season"] == season].copy()
        if len(df_s) > sample_size:
            df_s = df_s.sample(sample_size, random_state=42)

        features = features_dict[season] + ["lag_1"]
        X = df_s[features].values
        y = df_s["concentration"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        y_train_log = np.log1p(y_train)

        model = svm.SVR(kernel="rbf", C=c_val, epsilon=eps_val)
        model.fit(X_train_s, y_train_log)

        y_pred = np.expm1(model.predict(X_test_s))

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        summary.append(
            {"Season": season, "MAE": round(mae, 2), "R2": round(r2, 3)}
        )

        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="white")
        min_v, max_v = min(y_test.min(), y_pred.min()), max(
            y_test.max(), y_pred.max()
        )
        ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2)
        ax.set_title(f"{season}\nMAE={mae:.2f} | R²={r2:.3f}")
        ax.set_xlabel("Actual PM2.5")
        ax.set_ylabel("Predicted PM2.5")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Final SVR Performance Across Seasons", fontsize=20)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(summary)
