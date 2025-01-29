import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)


ROOT_PATH = os.getenv("ROOT_PATH")
DATASETS_PATH = f"{ROOT_PATH}{os.getenv("DATASETS_PATH")}"
SAVED_MODELS_PATH = f"{ROOT_PATH}{os.getenv("SAVED_MODELS_PATH")}"

DIGITS_DATASET = f"{DATASETS_PATH}{os.getenv("DIGITS_DATASET")}"
DIGITS_PROCESSED_DATASET_FILE = f"{DIGITS_DATASET}{os.getenv("DIGITS_PROCESSED_DATASET_FILE")}"

AND_SAVED_MODEL = f"{SAVED_MODELS_PATH}{os.getenv("AND_SAVED_MODEL")}"
XOR_SAVED_MODEL = f"{SAVED_MODELS_PATH}{os.getenv("XOR_SAVED_MODEL")}"
STUDENT_PASS_FAIL_SAVED_MODEL = f"{SAVED_MODELS_PATH}{os.getenv("STUDENT_PASS_FAIL_SAVED_MODEL")}"