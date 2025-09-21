import pandas as pd

def load_dataframe(csv_path: str, date_column: str = "Date", parse_dates: bool = True, index_as_date: bool = True) -> pd.DataFrame:
    if parse_dates:
        df = pd.read_csv(csv_path, parse_dates=[date_column])
    else:
        df = pd.read_csv(csv_path)
    if index_as_date:
        df.set_index(date_column, inplace=True)
    return df

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)