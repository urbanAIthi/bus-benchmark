import pandas as pd
import os
from tqdm import tqdm
from bus_benchmark.experiments.utils import preprocess_csv
from dotenv import load_dotenv

load_dotenv()

CSV_DATA_DIR = os.getenv("CSV_DATA_DIR", "")
PARQUET_DATA_DIR = os.getenv("PARQUET_DATA_DIR", "")

for csv_file in tqdm(os.listdir(CSV_DATA_DIR), desc="CSV files"):
    if csv_file.endswith(".csv"):
        print(f"Processing {csv_file}")
        df_csv = pd.read_csv(os.path.join(CSV_DATA_DIR, csv_file))
        df = preprocess_csv(df_csv)
        parquet_file = os.path.join(PARQUET_DATA_DIR, csv_file.replace('.csv', '.parquet'))
        df.to_parquet(parquet_file, index=False)
        print(f"Saved {parquet_file}")
