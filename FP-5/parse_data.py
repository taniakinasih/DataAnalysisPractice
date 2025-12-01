import pandas as pd

# LOAD DATA FILE
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# REMOVE MISSING PM2.5 CONCENTRATION
def remove_missing_pm(df: pd.DataFrame) -> pd.DataFrame:
    if "pm2.5" in df.columns:
        df = df.dropna(subset=["pm2.5"])
    return df
    
# CATEGORIZE THE MONTHS BASED ON SEASON
def add_season(df: pd.DataFrame) -> pd.DataFrame:
    def get_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["season"] = df["month"].apply(get_season)
    return df

#SELECT ONLY THE RELEVANT COLUMNS
def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["month", "pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Ir"]
    available = [c for c in cols if c in df.columns]
    return df[available]

#RENAME THE NAME OF RELEVANT COLUMNS
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "pm2.5": "concentration",
        "DEWP": "dew point",
        "TEMP": "temperature",
        "PRES": "pressure",
        "Iws": "wind speed",
        "Ir": "rainfall duration"
    }
    mapping = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=mapping)

def parse(path: str) -> pd.DataFrame:
    df = load_raw(path)
    df = remove_missing_pm(df)
    df = select_relevant_columns(df)
    df = rename_columns(df)
    df = add_season(df)  

    numeric_cols = df.select_dtypes(include="number").columns
    round_cols = [c for c in numeric_cols if c in ["concentration", "dew point", "temperature", "pressure", "wind speed", "rainfall duration"]]
    df[round_cols] = df[round_cols].round(2)

    return df


def get_descriptive_stats(df: pd.DataFrame):
    desired_cols = [
        "concentration", 
        "dew point", 
        "temperature", 
        "pressure", 
        "wind speed", 
        "rainfall duration"
    ]
    available_cols = [col for col in desired_cols if col in df.columns]
    return df[available_cols].describe()
