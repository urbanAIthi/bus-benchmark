import pandas as pd
import os
from tqdm import tqdm
from bus_benchmark import config
from bus_benchmark.experiments.utils import preprocess_csv

if not os.path.exists(config.PARQUET_DATA_DIR):
    os.makedirs(config.PARQUET_DATA_DIR)

for csv_file in tqdm(os.listdir(config.CSV_DATA_DIR), desc="CSV files"):
    if csv_file.endswith(".csv"):
        print(f"Processing {csv_file}")
        df_csv = pd.read_csv(os.path.join(config.CSV_DATA_DIR, csv_file))
        df = preprocess_csv(df_csv)
        parquet_file = os.path.join(config.PARQUET_DATA_DIR, csv_file.replace('.csv', '.parquet'))
        df.to_parquet(parquet_file, index=False)
        print(f"Saved {parquet_file}")