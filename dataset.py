import pandas as pd
import glob


files = glob.glob('/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data/train-*.parquet')
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

with open('/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data/input.txt', 'w', encoding='utf-8') as f:
    for text in df["input"]:
        if pd.notna(text):
            f.write(str(text).strip() + '\n')

with open('/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data/output.txt', 'w', encoding='utf-8') as f:
    for text in df["output"]:
        if pd.notna(text):
            f.write(str(text).strip() + '\n')
