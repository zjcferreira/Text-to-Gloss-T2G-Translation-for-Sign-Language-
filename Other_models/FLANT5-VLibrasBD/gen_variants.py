# Library Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import Counter
import json

def remove_junk(text):
    ret = str(text).replace('"', "")
    if 'BOF:' in ret:
        return ''
    return ret

def gloss_lower(text):
    return str(text).lower()

print("--- STARTING DATA PROCESSING ---")

# 1. Loads the original dataset
df = pd.read_csv('./VLibrasBD.csv') 
print(f"Initial total rows: {len(df)}")

VAR_PATH = './variants'
os.makedirs(VAR_PATH, exist_ok=True)

# 2. Cleaning
df['pgs'] = df['pgs'].apply(remove_junk)
# Ensures that pt and pgs are not null/empty
df.replace('', np.nan, inplace=True)
df.dropna(subset=['pgs', 'pt'], inplace=True) 

# ---- DUPLICATES REMOVAL ---
# Removal of rows where the Portuguese (pt) sentence is identical.
before_dedup = len(df)
df.drop_duplicates(subset=['pt'], inplace=True)
after_dedup = len(df)
print(f"Rows removed (duplicates): {before_dedup - after_dedup}")
print(f"Final total unique rows: {after_dedup}")

# Creates the lower column AFTER removing duplicates
df['pgs_lower'] = df['pgs'].apply(gloss_lower)

# 3. Data Split into Train / Validation / Test
# 80% Train, 10% Validation, 10% Test

# Separates 80% for Training and 20% for the rest (Temp)
train, temp = train_test_split(df, test_size=0.2, random_state=42)

# Splits the remaining 20% in half (10% Validation, 10% Test)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"Train Size: {len(train)}")
print(f"Validation Size: {len(val)}")
print(f"Test Size: {len(test)}")

# 4. Generation and saving of .csv files
variants = [['pgs', ''], ['pgs_lower', 'lower']]
# 3 splits in the dictionary
splits = {'train': train, 'val': val, 'test': test}

for variant_col, variant_name in variants:
    for split_name, data in splits.items():
        dset = data[['pt', variant_col]].copy()
        dset.columns = ['pt', 'pgs']
        # File name ex: pt-pgs.lower.train.csv, pt-pgs.lower.val.csv
        fname = f"pt-pgs.{variant_name}.{split_name}.csv" if variant_name else f"pt-pgs.{split_name}.csv"
        dset.to_csv(os.path.join(VAR_PATH, fname), index=False)
        print(f"File generated: {fname}")

# 5. Vocabulary files generation (based solely on training to avoid leakage)
with open(os.path.join(VAR_PATH, 'gloss_vocab.txt'), 'w') as f:
    # Only the TRAINING set is used to create the vocabulary. 
    # The model should not know words that only exist in the test set.
    gloss_vocab = Counter(train['pgs_lower'].str.cat(sep=' ').split())
    for gloss, count in gloss_vocab.most_common():
        f.write(f"{gloss}\n")

stats = {
    'n_glosses': sum(gloss_vocab.values()),
    'unique_glosses': len(gloss_vocab),
    'train_size': len(train),
    'val_size': len(val),
    'test_size': len(test)
}
with open(os.path.join(VAR_PATH, 'stats.json'), 'w') as f:
    json.dump(stats, f)

print("\nProcess completed successfully!")