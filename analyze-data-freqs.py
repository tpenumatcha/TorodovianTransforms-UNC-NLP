import pandas as pd
from pandarallel import pandarallel

MM_DATA_PATH = 'data/mm-prelabeled.csv'
FREQ_LABELS = ['no-transform', 'modal', 'intention', 'result', 'aspect', 'status', 'manner']

def getMMFreqs(input: pd.DataFrame) -> pd.DataFrame:
    freqDF = pd.DataFrame()
    
    def getSentenceFreqs(row) -> list:
        freqList = [0] * len(FREQ_LABELS)
        for c in row.mm_transforms.split(' '):
            freqList[int(c)] += 1
        return freqList
    
    freqDF[FREQ_LABELS] = input.parallel_apply(getSentenceFreqs, axis=1, result_type='expand')
    return freqDF

def isUnique(row) -> int:
    return 1 if len(set(row.mm_transforms.split(' '))) == 1 else 0

def main() -> None:
    mmInputDF = pd.read_csv(MM_DATA_PATH)
    numInstances = len(mmInputDF)
    mmFreqDF = getMMFreqs(mmInputDF)
    mmLabelCounts = {key: mmFreqDF[key].sum() for key in FREQ_LABELS} # total count for each label
    mmInstanceCounts = {key: len(mmFreqDF[mmFreqDF[key] > 0]) for key in FREQ_LABELS} # num of instances containing each label
    uniqueInstances = pd.DataFrame()
    uniqueInstances['isUnique'] = mmInputDF.parallel_apply(isUnique, axis=1)
    uniqueCount = len(uniqueInstances[uniqueInstances['isUnique'] == 0])
    
    print("\n=== Label breakdown (mm) ===")
    print(mmLabelCounts)
    transformCount = sum(mmLabelCounts[k] for k in mmLabelCounts.keys() if k != 'no-transform')
    print('Percentage breakdown (excluding non-transforms):')
    for key in mmLabelCounts.keys():
        if key != 'no-transform':
            print(f'{key}: {100*mmLabelCounts[key]/transformCount:.2f}% of labels')

    print("\n=== Instance breakdown (mm) ===")
    print(mmInstanceCounts)
    for key in mmInstanceCounts.keys():
        print(f"{key}: In {100*mmInstanceCounts[key]/numInstances:.2f}% of instances")

    print("\n=== Label breakdown (bin) ===")
    labelCount = sum(mmLabelCounts.values())
    binLabelCounts = {'no-transform': mmLabelCounts['no-transform'], 'transform': labelCount-mmLabelCounts['no-transform']}
    print(binLabelCounts)
    for key in binLabelCounts.keys():
        print(f'{key}: {100*binLabelCounts[key]/labelCount:.2f}% of labels')

    print("\n=== Instance breakdown (bin) ===")
    binInstanceCounts = {'no-transform': mmInstanceCounts['no-transform'], 'transform': sum(mmInstanceCounts[k] for k in mmInstanceCounts.keys() if k != 'no-transform')}
    print(binInstanceCounts)
    print(f"{100*binInstanceCounts['transform']/numInstances:.2f}% of instances contained a transform.")
    print(f"{100*uniqueCount/binInstanceCounts['transform']:.2f}% of instances containing a transform contained only one type of transform.")

    print(f"\nInstance count: {numInstances}")
    print(f"Label count: {labelCount}")

if __name__ == '__main__':
    pandarallel.initialize(verbose=0)
    main()
