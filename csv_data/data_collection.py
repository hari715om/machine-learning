import os
import sqlite3
import pandas as pd
import requests

from bs4 import BeautifulSoup  


# 1. DATABASE EXAMPLE 
def load_from_database(db_path="students.db"):

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            marks REAL
        )
    """)

    cur.execute("SELECT COUNT(*) FROM students")
    if cur.fetchone()[0] == 0:
        sample_students = [
            ("Amit", 20, 78.5),
            ("Priya", 21, 88.0),
            ("Rahul", 19, 67.0),
            ("Neha", 22, 92.5),
        ]
        cur.executemany("INSERT INTO students (name, age, marks) VALUES (?, ?, ?)", sample_students)
        conn.commit()

    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    return df


# 2. DATA WAREHOUSE EXAMPLE 
def load_from_datawarehouse(csv_path="sales_dw.csv"):

    if not os.path.exists(csv_path):
        
        data = {
            "order_id": [1, 2, 3, 4],
            "customer": ["C1", "C2", "C1", "C3"],
            "amount": [500.0, 1200.5, 300.0, 750.0],
            "city": ["Delhi", "Mumbai", "Delhi", "Bengaluru"],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    return df


# 3. EXPERIMENT LOGS EXAMPLE 
def load_from_experiment_logs(log_path="experiment_logs.csv"):

    if not os.path.exists(log_path):
        data = {
            "experiment_id": [1, 1, 2, 2],
            "model_name": ["baseline", "improved", "baseline", "improved"],
            "accuracy": [0.78, 0.82, 0.80, 0.85],
            "timestamp": [
                "2025-12-01 10:00:00",
                "2025-12-01 11:00:00",
                "2025-12-02 09:30:00",
                "2025-12-02 10:15:00",
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(log_path, index=False)

    df = pd.read_csv(log_path)
    return df


# 4. CLOUD STORAGE EXAMPLE 
def load_from_cloud_s3(bucket_name="my-demo-bucket", object_key="cloud_data.csv"):

    try:
        import boto3  
    except ImportError:
        print("boto3 not installed, skipping S3 example.")
        return None

    try:
        s3 = boto3.client("s3")
        local_path = "cloud_data_downloaded.csv"
        s3.download_file(bucket_name, object_key, local_path)

        df = pd.read_csv(local_path)
        return df
    except Exception as e:
        print(f"Could not load from S3 (check credentials/bucket): {e}")
        return None


# 5. WEB SCRAPING EXAMPLE
def load_from_web_scraping(url="https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)"):

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching page: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    if table is None:
        print("No table found on the page.")
        return None

    headers = []
    for th in table.find_all("th"):
        headers.append(th.get_text(strip=True))

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td"])
        if not cells:
            continue
        row = [c.get_text(strip=True) for c in cells]
        rows.append(row)

    if not rows:
        print("No data rows found in table.")
        return None

    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]
    headers = (headers + [f"col_{i}" for i in range(len(headers), max_len)])[:max_len]

    df = pd.DataFrame(rows, columns=headers)
    return df.head(10)  


# 6. REST API EXAMPLE
def load_from_api(url="https://api.github.com/repos/pandas-dev/pandas"):

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

    data = response.json()
    subset = {
        "full_name": data.get("full_name"),
        "stargazers_count": data.get("stargazers_count"),
        "forks_count": data.get("forks_count"),
        "open_issues": data.get("open_issues"),
        "watchers": data.get("watchers"),
    }
    df = pd.DataFrame([subset])
    return df


def main():
    print("\n1. DATA FROM DATABASE")
    db_df = load_from_database()
    print(db_df, "\n")

    print("2. DATA FROM DATA WAREHOUSE CSV ")
    dw_df = load_from_datawarehouse()
    print(dw_df, "\n")

    print("3. DATA FROM EXPERIMENT LOGS ")
    exp_df = load_from_experiment_logs()
    print(exp_df, "\n")

    print("4. DATA FROM CLOUD STORAGE ")
    cloud_df = load_from_cloud_s3()
    if cloud_df is not None:
        print(cloud_df.head(), "\n")
    else:
        print("Skipped S3 example (not configured).\n")

    print("5. DATA FROM WEB SCRAPING ")
    web_df = load_from_web_scraping()
    if web_df is not None:
        print(web_df, "\n")
    else:
        print("Web scraping example failed or skipped.\n")

    print("6. DATA FROM PUBLIC REST API ")
    api_df = load_from_api()
    if api_df is not None:
        print(api_df, "\n")
    else:
        print("API example failed or skipped.\n")


if __name__ == "__main__":
    main()
