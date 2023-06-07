import pandas as pd

# 70/30 training/test split
TRAINING_N = 70_000
TEST_N = 30_000

STATE_SEED = 2022

BIN_INPUT_PATH = 'data/bin-prelabeled.csv'
MM_INPUT_PATH = 'data/mm-prelabeled.csv'

TRAINING_SET_PREFIX = 'data/training-set'
TEST_SET_PREFIX = 'data/test-set'

if __name__ == '__main__':
    binDF = pd.read_csv(BIN_INPUT_PATH)[['instance', 'file-name', 'bin_transforms', 'review-id']]
    mmDF = pd.read_csv(MM_INPUT_PATH)[['instance', 'file-name', 'mm_transforms', 'review-id']]
    
    binDF = binDF.sample(frac=1, random_state=STATE_SEED) # shuffle DF
    binTrainingDF = binDF.head(TRAINING_N) # get first TRAINING_N rows
    binTestDF = binDF.tail(TEST_N) # get last TEST_N rows
    binTrainingDF.to_csv(f"{TRAINING_SET_PREFIX}-bin.csv")
    binTestDF.to_csv(f"{TEST_SET_PREFIX}-bin.csv")

    mmDF = mmDF.sample(frac=1, random_state=STATE_SEED)
    mmTrainingDF = mmDF.head(TRAINING_N)
    mmTestDF = mmDF.tail(TEST_N)
    mmTrainingDF.to_csv(f"{TRAINING_SET_PREFIX}-mm.csv")
    mmTestDF.to_csv(f"{TEST_SET_PREFIX}-mm.csv")
