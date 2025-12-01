import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress, zscore
from IPython.display import display, HTML
import statsmodels.api as sm 
import numpy as np 

# ---------------------------------------------------------------
# REMOVE OUTLIERS AND CATEGORIZE POLLUTION CONCENTRATION FUNCTION
# --------------------------------------------------------------

def remove_outliers(df, col="concentration", threshold=3):
    """Removes outliers from a specified column using Z-score (absolute Z-score <= threshold)."""
    df = df.copy()
    if col not in df.columns or df[col].std() == 0:
        return df
    # Filter for data points where the absolute Z-score is less than or equal to the threshold
    df_clean = df[np.abs(zscore(df[col])) <= threshold].copy()
    return df_clean

def add_pollution_category(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'pollution_level' categorical column based on PM2.5 concentration bins (WHO/AQI limits)."""
    # PM2.5 AQI limits (approximate): 0-12 good, 12-35.4 moderate, 35.4-55.4 unhealthy, >55.4 very unhealthy
    bins = [-np.inf, 12.0, 35.4, 55.4, np.inf]
    labels = ["Good", "Moderate", "Unhealthy", "Very Unhealthy"]
    if "concentration" in df.columns:
        df["pollution_level"] = pd.cut(df["concentration"], bins=bins, labels=labels, right=True, include_lowest=True)
    return df

# ----------------------------------------------------------
# 1. LINEAR REGRESSION PER SEASON (Simple Regression)
# ----------------------------------------------------------

def regression(df, dep_var="concentration", exclude_vars=None):
    """Performs simple linear regression (one independent variable) for each season."""
    results = {}

    if exclude_vars is None:
        exclude_vars = ["month"]
        
    exclude_vars_lower = [v.lower() for v in exclude_vars + ["season"]]
    
    # Remove Outliers
    df_clean = remove_outliers(df, col=dep_var) 

    if "season" not in df_clean.columns:
        print("Error: 'season' column not found in DataFrame.")
        return pd.DataFrame()
    
    seasons = df_clean["season"].unique()
    
    # Loop through seasons
    for season in seasons:
        df_s = df_clean[df_clean["season"] == season]
        
        # Select numeric columns that are not the dependent variable or in the exclude list
        independent_vars = [c for c in df_s.select_dtypes(include="number").columns 
                            if c != dep_var and c.lower() not in exclude_vars_lower]

        if not independent_vars:
            continue
            
        data = []
        for var in independent_vars:
            # Check for sufficient data points and non-zero variance
            if len(df_s[var]) < 2 or df_s[var].std() == 0:
                continue

            # Linear Regression using scipy.stats.linregress
            slope, intercept, r_value, p_value, std_err = linregress(df_s[var], df_s[dep_var])
            data.append({
                "Independent Variable": var,
                "Slope": round(slope, 4),
                "Intercept": round(intercept, 4),
                "R": round(r_value, 4),
                "R^2": round(r_value**2, 4),
                "P-value": round(p_value, 4),
                "Std Err": round(std_err, 4)
            })

        summary = pd.DataFrame(data).set_index("Independent Variable")
        display(HTML(f"<h3>Linear Regression Summary - {season}</h3>"))
        display(summary)
        results[season] = summary

    return results

# ------------------------------------------------------------------------
# 2. LINEAR REGRESSION FOCUSING ON HIGH CONCENTRATION OF PM2.5 PER SEASON
# ------------------------------------------------------------------------

def regression_high_concentration(df: pd.DataFrame, high_levels=["Unhealthy", "Very Unhealthy"]):
    """Performs simple linear regression for high pollution levels only, per season."""
    df = add_pollution_category(df)
    df_high = df[df["pollution_level"].isin(high_levels)].copy()
    df_high_clean = remove_outliers(df_high, col="concentration") 

    if "season" not in df_high_clean.columns:
        print("Error: 'season' column not found in DataFrame.")
        return pd.DataFrame()
    
    seasons = df_high_clean["season"].unique()
    # Define primary independent variables for meteorological analysis
    independent_vars = ["dew point", "temperature", "pressure", "wind speed", "rainfall duration"]
    independent_vars = [v for v in independent_vars if v in df_high_clean.columns]

    all_results = []

    for season in seasons:
        df_s = df_high_clean[df_high_clean["season"] == season]
        for var in independent_vars:
            # Check for sufficient data points and non-zero variance
            if len(df_s) > 1 and df_s[var].std() > 0:
                slope, intercept, r_value, p_value, std_err = linregress(df_s[var], df_s["concentration"])
                row = {
                    "Season": season,
                    "Independent Variable": var,
                    "Slope": round(slope, 4),
                    "Intercept": round(intercept, 4),
                    "R": round(r_value, 4),
                    "R^2": round(r_value**2, 4),
                    "P-value": round(p_value, 4),
                    "Std Err": round(std_err, 4)
                }
            else:
                # Indicate insufficient data/variance
                row = {
                    "Season": season,
                    "Independent Variable": var,
                    "Slope": None, "Intercept": None, "R": None, 
                    "R^2": None, "P-value": None, "Std Err": None
                }
            all_results.append(row)

    if not all_results:
        print("No valid data found for high concentration regression.")
        return pd.DataFrame()
    
    summary = pd.DataFrame(all_results)
    summary.set_index(["Season", "Independent Variable"], inplace=True)
    display(HTML("<h3>Linear Regression Summary - High Concentration Only</h3>"))
    display(summary)
    return summary

# --------------------------------------------------------------------------------
# 3. MULTIPLE LINEAR REGRESSION FOR COMPARING EACH CATEGORY OF PM2.5 CONCENTRATION
# --------------------------------------------------------------------------------

def multiple_regression_comparison(df: pd.DataFrame):
    """Performs multiple linear regression for each pollution level category against all weather variables."""
    
    df = add_pollution_category(df)
    df_clean = remove_outliers(df, col="concentration")
    
    levels = df_clean["pollution_level"].unique()
    results = []
    
    X_vars = ["dew point", "temperature", "pressure", "wind speed", "rainfall duration"]
    X_vars = [v for v in X_vars if v in df_clean.columns]
    Y_var = "concentration"
    
    if not X_vars:
        print("Independent variables not found.")
        return pd.DataFrame()

    for level in levels:
        df_l = df_clean[df_clean["pollution_level"] == level].copy()
        

        # Need N > k + 1 observations for k variables
        if len(df_l) < len(X_vars) + 1:
            continue
            
        X = df_l[X_vars]
        y = df_l[Y_var]
        X = sm.add_constant(X) # Add intercept term
        
        # Multiple Regression Using OLS
        try:
            model = sm.OLS(y, X, missing='drop').fit()
        except Exception:
            continue
        
        results.append({
            "Pollution Level": level,
            "N": len(model.resid), # Actual N used after dropping NaNs
            "R-squared": round(model.rsquared, 4),
            "Adjusted R-squared": round(model.rsquared_adj, 4),
            "F-statistic P-value": round(model.f_pvalue, 4),
        })

    summary = pd.DataFrame(results).set_index("Pollution Level")
    display(HTML("<h3>Multiple Regression (PM2.5 vs. Combination of All Variables)</h3>"))
    display(summary)

    return summary

# --------------------------------------------------------------------------------------------------------------
# 4. CONTEXTUAL MULTIPLE REGRESSION (Model Selection Based on Previous Analysis)
# --------------------------------------------------------------------------------------------------------------

# ------------------------------------------
# MODEL SETS (Based on assumed feature selection results)
# ------------------------------------------

MODEL_SETS = {
    "A": {
        "Winter": ["dew point", "pressure", "wind speed"],
        "Spring": ["dew point", "temperature", "pressure", "wind speed"],
        "Summer": ["dew point", "pressure", "rainfall duration"],
        "Autumn": ["dew point", "temperature", "wind speed", "rainfall duration"]
    },
    "B": {
        "Winter": ["dew point", "pressure", "wind speed", "temperature"],
        "Spring": ["dew point", "wind speed"],
        "Summer": ["dew point", "pressure", "rainfall duration"],
        "Autumn": ["temperature", "wind speed", "pressure"]
    },
    "C": {
        "Winter": ["dew point", "pressure", "wind speed", "temperature", "rainfall duration"],
        "Spring": ["dew point", "wind speed", "pressure"],
        "Summer": ["dew point", "pressure", "rainfall duration", "temperature", "wind speed"],
        "Autumn": ["dew point", "temperature", "wind speed", "pressure"]
    },
    "D": {
        "Winter": ["temperature", "wind speed", "pressure"],
        "Spring": ["wind speed", "temperature"],
        "Summer": ["rainfall duration", "temperature"],
        "Autumn": ["dew point", "pressure", "wind speed"]
    },
    # Set E is often a full model comparison
    "E": {
        "Winter": ["dew point", "pressure", "wind speed", "temperature", "rainfall duration"],
        "Spring": ["dew point", "temperature", "pressure", "wind speed"],
        "Summer": ["dew point", "pressure", "rainfall duration", "temperature", "wind speed"],
        "Autumn": ["dew point", "temperature", "wind speed", "rainfall duration"]
    }
}

# ------------------------------------------
# MULTIPLE REGRESSION CORE FUNCTION (Internal)
# ------------------------------------------

def _run_contextual_regression(df: pd.DataFrame, custom_models: dict, title_suffix: str,
                              high_pollution_only: bool = False, dep_var="concentration"):
    """Internal core function to run multiple regression based on custom model sets per season."""
    df = add_pollution_category(df)
    
    # Filter data based on pollution level requirement
    if high_pollution_only:
        high_levels = ["Unhealthy", "Very Unhealthy"]
        df_filtered = df[df["pollution_level"].isin(high_levels)].copy()
        df_clean = remove_outliers(df_filtered, col=dep_var)
    else:
        df_clean = remove_outliers(df, col=dep_var)

    if "season" not in df_clean.columns:
        print("Error: 'season' column not found in DataFrame.")
        return pd.DataFrame(), {}

    seasons = custom_models.keys()
    all_results = []
    r_squared_n_map = {}

    for season in seasons:
        X_vars = custom_models.get(season, [])
        df_s = df_clean[df_clean["season"] == season].copy()
        valid_X_vars = [v for v in X_vars if v in df_s.columns]

        current_n_initial = len(df_s)
        # Check for minimum number of observations (N > k + 1)
        if len(df_s) < len(valid_X_vars) + 1 or not valid_X_vars:
            r_squared_n_map[season] = {"R-squared": 0, "N": current_n_initial}
            continue

        X = df_s[valid_X_vars]
        y = df_s[dep_var]
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X, missing='drop').fit()
            current_rsquared = round(model.rsquared, 4)
            current_n = len(model.resid) # Actual N used after dropping NaNs

            all_results.append({
                "Season": season,
                "Model Variables": ", ".join(valid_X_vars),
                "N": current_n,
                "R-squared": current_rsquared,
                "Adjusted R-squared": round(model.rsquared_adj, 4),
                "P-value": round(model.f_pvalue, 4)
            })

            r_squared_n_map[season] = {"R-squared": current_rsquared, "N": current_n}
        except Exception:
            r_squared_n_map[season] = {"R-squared": 0, "N": current_n_initial}

    summary = pd.DataFrame(all_results).set_index("Season")
    data_scope = "High Pollution Only" if high_pollution_only else "All Data"
    # This is the line where the f-string error occurred in the traceback, now verified as correct.
    display(HTML(f"<h3>Contextual Multiple Regression Summary ({title_suffix} - {data_scope})</h3>"))
    display(summary)

    return summary, r_squared_n_map

# --------------------------------------------------------
# 5. PUBLIC API FOR CONTEXTUAL REGRESSION 
# --------------------------------------------------------

def run_contextual_regression(df: pd.DataFrame, set_name: str, high_pollution: bool = False, dep_var="concentration"):
    """Runs multiple regression for a specified model set (A-E)."""
    set_name = set_name.upper()
    custom_models = MODEL_SETS.get(set_name)
    if not custom_models:
        print(f"Error: Model set '{set_name}' not found.")
        return pd.DataFrame()

    title = f"SET {set_name}"
    # Returns only the summary DataFrame [0]
    return _run_contextual_regression(df, custom_models, title, high_pollution, dep_var)[0]

# Legacy wrappers for specific model sets (e.g., Set E)
contextual_multiple_regression_5 = lambda df: run_contextual_regression(df, "E", high_pollution=False)
contextual_multiple_regression_high_pollution_5 = lambda df: run_contextual_regression(df, "E", high_pollution=True)


# -----------------------------
# 6. PLOTTING FUNCTIONS (Set E)
# -----------------------------

def plot_predicted_vs_actual_set_5(df: pd.DataFrame, dep_var="concentration", high_pollution_only: bool = False):
    """Plots Predicted vs. Actual PM2.5 based on Multiple Regression Model Set E, per season."""
    custom_models = MODEL_SETS.get("E")
    
    # Run core regression logic to get R^2 and N values for plotting titles
    _, r_squared_n_map = _run_contextual_regression(df, custom_models, "Set E", high_pollution_only=high_pollution_only, dep_var=dep_var)
    
    if high_pollution_only:
        title = "Predicted vs. Actual PM2.5 - Model Set E (HIGH POLLUTION ONLY)"
        scatter_color = '#D9534F' # Reddish color for high pollution
        high_levels = ["Unhealthy", "Very Unhealthy"]
        df_filtered = df[df["pollution_level"].isin(high_levels)].copy()
    else:
        title = "Predicted vs. Actual Concentration (PM2.5) using Model Set E (ALL DATA)"
        scatter_color = '#5CB85C' # Greenish color for all data
        df_filtered = df.copy()

    df_clean = remove_outliers(df_filtered, col=dep_var) 

    seasons = custom_models.keys()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten() 
    
    plt.suptitle(title, fontsize=16, y=1.03)

    for i, season in enumerate(seasons):
        ax = axes[i]
        X_vars = custom_models.get(season, []) 
        df_s = df_clean[df_clean["season"] == season].copy()
        valid_X_vars = [v for v in X_vars if v in df_s.columns]
        
        current_rsquared = r_squared_n_map.get(season, {}).get("R-squared", 0)
        current_n = r_squared_n_map.get(season, {}).get("N", len(df_s)) 
        
        # Check if regression was successful (based on R^2/N map)
        if not valid_X_vars or current_n < len(valid_X_vars) + 1:
            ax.set_title(f"{season} (Insufficient Data)", color='red')
            ax.text(0.5, 0.5, f"Observations: {current_n}", ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel("Actual PM2.5", fontsize=10)
            ax.set_ylabel("Predicted PM2.5", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            continue
            
        X = df_s[valid_X_vars]
        y = df_s[dep_var]
        X = sm.add_constant(X) 
        
        try:
            model = sm.OLS(y, X, missing='drop').fit()
            predictions = model.predict(X)
            y_used = y[model.resid.index] # Actual values used in the model
            
            ax.scatter(y_used, predictions, alpha=0.6, color=scatter_color) 
            
            # Add y=x line for ideal fit
            min_val = min(y_used.min(), predictions.min())
            max_val = max(y_used.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Fit (y=x)')
            
            ax.set_title(f"{season} ($R^2$: {current_rsquared:.3f}, N={current_n})", fontsize=14)
            ax.set_xlabel("Actual PM2.5", fontsize=10)
            ax.set_ylabel("Predicted PM2.5", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper left')

        except Exception:
            ax.set_title(f"{season} (Regression Error)", color='red')

    plt.tight_layout()
    plt.show()

# Wrapper for high pollution plot compatibility
plot_predicted_vs_actual_high_pollution_set_5 = lambda df, dep_var="concentration": plot_predicted_vs_actual_set_5(df, dep_var, high_pollution_only=True)


#---------------------------------------
# HEATMAP
#--------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, dep_var="concentration"):
    """
    Generates and displays a heatmap of the Pearson correlation coefficients 
    between PM2.5 concentration and all meteorological variables.
    """
    
    # 1. Define Variables
    X_vars = ["dew point", "temperature", "pressure", "wind speed", "rainfall duration"]
    
    # Combine all variables, including the dependent variable
    all_vars = [dep_var] + X_vars
    
    # Ensure only columns present in the DataFrame are used
    valid_vars = [v for v in all_vars if v in df.columns]

    if len(valid_vars) < 2:
        print("Error: DataFrame must contain 'concentration' and at least one weather variable.")
        return

    # 2. Clean Data (Removing outliers based on concentration)
    # Using the 'remove_outliers' function you defined earlier
    df_clean = remove_outliers(df, col=dep_var)
    
    # 3. Calculate the Correlation Matrix
    # Calculate correlation only for the valid variables
    correlation_matrix = df_clean[valid_vars].corr(numeric_only=True)
    
    # 4. Plot the Heatmap
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap using seaborn
    sns.heatmap(
        correlation_matrix,
        annot=True,              # Display the coefficient values in the cells
        fmt=".2f",               # Format numbers to 2 decimal places
        cmap='coolwarm',         # Color scheme (coolwarm is good for positive/negative correlation)
        linewidths=.5,           # Lines between cells
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )
    
    # Title and Adjustments
    plt.title('Correlation Heatmap: PM2.5 Concentration and Meteorological Variables', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# --- HOW TO USE THIS FUNCTION ---

# Make sure your 'df' DataFrame is loaded and the 'remove_outliers' 
# function from your full code block has been run previously.

# Example Call:
# plot_correlation_heatmap(df)