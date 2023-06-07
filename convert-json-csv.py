import pandas as pd
import json

JSON_PATH = "data/scraped-results.jsonl"
OUT_PATH = "data/scraped-results.csv"

def import_data(path: str):
    with open(path, 'r') as f:
        input = json.load(f)
    input_df = pd.DataFrame.from_dict(input)
    return input_df

if __name__ == '__main__':
    df = import_data(JSON_PATH)
    df.to_csv(OUT_PATH)
    