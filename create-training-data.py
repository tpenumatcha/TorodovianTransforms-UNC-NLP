import sys
import pandas as pd
import ast
import ctypes

OUTPUT_PATH = 'data/unlabeled-training-data.csv'

STATE_SEED = 2022

def process(file_name: str) -> list:
    review_df = pd.read_csv(file_name)
    out_list = []
    for row_ind in review_df.index:
        out_list.extend({'instance': s, 'file-name': file_name} for s in ast.literal_eval(review_df['text'][row_ind]) if s != '')
    return out_list


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('ERROR: Please provide at least one CSV file.')
        exit()
    instances = []
    for ind, file in enumerate(sys.argv[1:]):
        if '.' not in file or file.split('.')[1] != 'csv':
            print('ERROR: All provided files must have the CSV extension.')
            exit()
        print(f"Processing file {ind+1}/{len(sys.argv)-1}...")
        instances.extend(process(file))
    outputDF = pd.DataFrame.from_records(instances).drop_duplicates(subset=['instance'], keep='first')
    simplehash = lambda row : ctypes.c_uint64(hash(row.instance)).value.to_bytes(8, "big").hex()
    print("Creating hashed review IDs...")
    outputDF['review-id'] = outputDF.apply(simplehash, axis=1)
    print("Shuffling...")
    outputDF = outputDF.sample(frac=1, random_state=STATE_SEED)
    print(f"Writing to {OUTPUT_PATH}...")
    outputDF.to_csv(OUTPUT_PATH)
    print("Finished!")
        