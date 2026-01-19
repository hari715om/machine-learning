import pandas as pd

CSV_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
EXCEL_PATH = r"C:\Users\KIIT0001\Downloads\SampleData\SampleData.xlsx"


def load_remote_csv(url: str) -> pd.DataFrame:

    print(f"\n[REMOTE CSV] Downloading from: {url}")
    df = pd.read_csv(url)
    print("[REMOTE CSV] Shape:", df.shape)
    print("[REMOTE CSV] Columns:", list(df.columns)[:10], "...")
    print("\n[REMOTE CSV] Head:")
    print(df.head(5))
    return df


def load_local_excel(filepath: str, sheet_name: str = None) -> pd.DataFrame:

    print(f"\n[LOCAL EXCEL] Loading file from: {filepath}")

    try:
        df = pd.read_excel(filepath, sheet_name="SalesOrders")
    except Exception as e:
        print(f"[LOCAL EXCEL] Error loading file: {e}")
        return None

    print("[LOCAL EXCEL] Shape:", df.shape)
    print("[LOCAL EXCEL] Columns:", list(df.columns))
    
    print("\n[LOCAL EXCEL] First 5 rows:")
    print(df.head(5))

    return df


def main():
    print("REMOTE CSV & EXCEL DATA COLLECTION ")

    csv_df = load_remote_csv(CSV_URL)

    df = load_local_excel(EXCEL_PATH)

    print("\n[CSV] Example feature subset :")
    csv_features = csv_df[["Pclass", "Sex", "Age", "Fare"]].head(10)
    print(csv_features)

    if df is not None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).head(5)
        print("\nNumerical feature sample:")
        print(numeric_cols)


if __name__ == "__main__":
    main()
