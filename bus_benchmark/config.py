from dotenv import load_dotenv
import os

load_dotenv()

ROOT_PATH = os.getenv('ROOT_PATH')
assert ROOT_PATH is not None, "ROOT_PATH environment variable is not set"

DATA_DIR = ROOT_PATH + '/data/'
CSV_DATA_DIR = DATA_DIR + "csv_data/"
PARQUET_DATA_DIR = DATA_DIR + "parquet_data/"

WANDB_ENTITY = "urbanai"
WANDB_PROJECT = "bus-benchmark-philip"
