import pandas as pd


d = {
    'row1': {
        'A': 1, 'B': 2, 'C': 3
    },
    'row2': {
        'A': 4, 'B': 5, 'C': 6
    },
    'row3': {
        'A': 7, 'B': 8, 'C': 9
    }
}

df = pd.DataFrame.from_dict(d, orient='index')

for row in df.itertuples():
    print(row)
    print(f"Index: {row.Index}, A: {row.A}, B: {row.B}, C: {row.C}")