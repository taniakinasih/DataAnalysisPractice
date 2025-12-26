import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, zscore, pearsonr
from IPython.display import display, HTML


# OUTLIER REMOVAL FUNCTION

def remove_outliers(df, col="concentration", threshold=3):
    """
    Remove outliers using Z-score threshold.
    """
    df = df.copy()
    df[f"z_{col}"] = zscore(df[col])
    df_clean = df[df[f"z_{col}"].abs() <= threshold].drop(columns=[f"z_{col}"])
    return df_clean


# DESCRIPTIVE STATISTICS

def get_descriptive_stats(df):

    cols = [
        "concentration",
        "dew point",
        "temperature",
        "pressure",
        "wind speed",
        "rainfall duration"
    ]

    # CENTRAL TENDENCY
    central = pd.DataFrame({
        "mean": df[cols].mean(),
        "median": df[cols].median(),
        "mode": df[cols].mode().iloc[0]
    }).T.round(2)

    display(HTML("<h3>Table 1. Central Tendency Summary Statistics</h3>"))
    display(central)

    # DISPERSION
    dispersion = pd.DataFrame({
        "std": df[cols].std(),
        "min": df[cols].min(),
        "max": df[cols].max(),
        "range": df[cols].max() - df[cols].min(),
        "25th": df[cols].quantile(0.25),
        "75th": df[cols].quantile(0.75),
        "IQR": df[cols].quantile(0.75) - df[cols].quantile(0.25)
    }).T.round(2)

    display(HTML("<h3>Table 2. Dispersion Summary Statistics</h3>"))
    display(dispersion)

    return central, dispersion



# SCATTER PLOTS PER SEASON

def scatter_by_season(df):

    df = remove_outliers(df, "concentration")

    variables = ["dew point", "temperature", "pressure", "wind speed", "rainfall duration"]
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    color_map = {
        "dew point": "blue",
        "temperature": "red",
        "pressure": "green",
        "wind speed": "yellow",
        "rainfall duration": "black"
    }

    for var in variables:

        plt.figure(figsize=(18, 4))
        plt.suptitle(f"{var.capitalize()} vs PM2.5 Across Seasons (Outliers Removed)", fontsize=14)

        for i, season in enumerate(seasons):

            temp = df[df["season"] == season]

            # regression
            slope, intercept, r_value, _, _ = linregress(temp[var], temp["concentration"])

            plt.subplot(1, 4, i + 1)
            color = color_map[var]

            plt.scatter(temp[var], temp["concentration"], s=8, alpha=0.4, color=color)
            plt.plot(temp[var], intercept + slope * temp[var], color=color)

            plt.title(f"{season}\nr = {r_value:.3f}")
            plt.xlabel(var.capitalize())
            plt.ylabel("PM2.5 Concentration")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# TABLE SUMMARY OF CORRELATIONS
def table_summary(df, num=3):

    df_clean = remove_outliers(df, "concentration")

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    variables = [
        ("dew point", "Dew Point"),
        ("temperature", "Temperature"),
        ("pressure", "Pressure"),
        ("wind speed", "Wind Speed"),
        ("rainfall duration", "Rainfall Duration")
    ]

    data = []

    for var, label in variables:
        row = {"Variable": label}

        for season in seasons:
            df_s = df_clean[df_clean["season"] == season]

            if len(df_s) > 1:
                r, _ = pearsonr(df_s[var], df_s["concentration"])
                row[season] = round(r, 3)
            else:
                row[season] = None

        data.append(row)

    summary = pd.DataFrame(data).set_index("Variable")
    summary["Strongest Correlation"] = summary.abs().idxmax(axis=1)

    display(HTML(f"<h3>Table {num}. Summary of Correlation (r) Between PM2.5 and Meteorological Variables</h3>"))
    display(summary)

    return summary
