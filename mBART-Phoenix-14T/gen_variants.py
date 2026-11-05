# DATA PREPROCESSING: ENSURES DATA REACHES mBART CLEAN AND STANDARDIZED

import pandas as pd         # Manipulate spreadsheets
import numpy as np          # Handle empty/null numbers -> Used to reference NaNs (np.nan)
import os                   # Create folders and save files
from collections import Counter # Count vocabulary
import json                 # Save configurations

# --- CONFIGURATIONS ---
# Dictionary creation (denoted by {} brackets) using Key:Value schema. Makes code more elegant; to open the test file, for instance, just call INPUT_FILES['test']
INPUT_FILES = {
    'train': './PHOENIX_TRAIN.csv',
    'val':   './PHOENIX_DEV.csv',    
    'test':  './PHOENIX_TEST.csv'
}

# Output configuration
VAR_PATH = './variants'
SRC_COL = 'de'   # Text column name (German)
TGT_COL = 'dgs'  # Gloss column name (DGS)

def remove_junk(text):                # QUOTE REMOVER
    ret = str(text).replace('"', "")  # TAKES THE TEXT AND REMOVES QUOTES. EX: "CAR" -> CAR
    if 'BOF:' in ret:                 # IF IT FINDS AN EXTRACTION FLAW ('BOF'), TREATS IT AS ELECTRONIC JUNK AND REMOVES IT, RETURNING EMPTY '' 
        return ''
    return ret

def gloss_lower(text):               # Converts everything to lowercase, since for NLP models, CAR, Car, and car are different tokens.
    return str(text).lower()

def process_dataset(name, filepath): # Main function, receives the Excel file name and its location (path)
    print(f"\n--- Processing: {name.upper()} ---")
    
    # 1. Loading 
    df = pd.read_csv(filepath)       # Loads into dataframe format
    
    print(f"Initial total: {len(df)}")
    
    # 2. Cleaning
    df[TGT_COL] = df[TGT_COL].apply(remove_junk)      # For each element in the gloss column, removes quotes (.apply applies the function to all elements)
    df[SRC_COL] = df[SRC_COL].apply(lambda x: str(x)) # For each element in the German sentences column, transforms sentences into strings
    
    # Remove empty rows
    df.replace('', np.nan, inplace=True)               # Pandas struggles to delete rows with ''. Thus, converts to NaN
    initial_len = len(df)                              # Calculates initial length, before deleting empty rows
    df.dropna(subset=[TGT_COL, SRC_COL], inplace=True) # Drops NaN rows in the gloss and German sentences columns
    if len(df) < initial_len:                             
        print(f"Removed {initial_len - len(df)} empty records.") # Prints how many rows were removed, if any
    
    # 3. Create Lower variant
    df[f'{TGT_COL}_lower'] = df[TGT_COL].apply(gloss_lower)  # Creates a new column (dgs_lower) which is simply the dgs column in lowercase letters
    
    return df                                                # Returns the dataframe (df) with changes applied

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

print("--- STARTING DATA PROCESSING (PHOENIX DATASET) ---")
os.makedirs(VAR_PATH, exist_ok=True) # Creates the './variants' folder defined in VAR_PATH. If it already exists, runs code without complaining (exist_ok=True)

splits = {} # Initially empty dictionary

# Processes the 3 files using the function
for split_name, file_path in INPUT_FILES.items():                # INPUT_FILES.items -> dictionary files: 'train','val', and 'test'
    splits[split_name] = process_dataset(split_name, file_path)  # Processes each file, placing it in the splits dictionary. Ex: split_name='train' and  
                                                                 # file_path='./PHOENIX_TRAIN.csv' to handle the training spreadsheet.
    
# 4. Generating and saving .csv files ready for training
# variants = [[Original Column, File Suffix]]
variants = [[TGT_COL, ''], [f'{TGT_COL}_lower', 'lower']]  # Creates a two-position array: dgs (no suffix) and dgs_lower (lower suffix)

print("\n--- Saving Files ---")
for variant_col, variant_suffix in variants:            # Loop that iterates over the variants (dgs and dgs_lower) and over the datasets (train, val, and test)
    for split_name, data in splits.items():
        # Selects only necessary columns
        dset = data[[SRC_COL, variant_col]].copy()      # Creates the dataset, de and dgs_lower (or dgs) columns, using the .copy command
        
        # Renames to 'de' and 'dgs' to match the training script
        dset.columns = ['de', 'dgs'] 
        
        # File name: de-dgs.lower.train.csv or de-dgs.train.csv. Ex: de-dgs.lower.train.csv (if it has a suffix) or de-dgs.train.csv (no suffix)
        fname = f"de-dgs.{variant_suffix}.{split_name}.csv" if variant_suffix else f"de-dgs.{split_name}.csv"
        
        # Removes extra dot if there is no suffix (ex: ..csv -> .csv)
        fname = fname.replace('..', '.') 
        
        save_path = os.path.join(VAR_PATH, fname)           # Saves path by concatenating VAR_PATH = './variants' and fname=de-dgs.train.csv
        dset.to_csv(save_path, index=False)                 # Saves spreadsheets to csv. Index=False prevents saving the code's 0,1,2,... indices
        print(f"Saved: {fname}")

# 5. Vocabulary files generation (Based on TRAINING only)
print("\n--- Generating Vocabulary ---")
train_df = splits['train']             # Creates a dataframe exclusively with training values
# Takes the training lower column
gloss_list = train_df[f'{TGT_COL}_lower'].str.cat(sep=' ').split()      # .str.cat -> grabs all rows and generates a single string, using whitespace as separator || .split -> separates all words in this giant string to analyze vocabulary
gloss_vocab = Counter(gloss_list) # Counter reads this giant list and counts word (gloss) frequency. Ex: {"car": 50, "house": 20, "me":300} 

with open(os.path.join(VAR_PATH, 'gloss_vocab.txt'), 'w') as f: # Opens a "gloss_vocab.txt" file and uses .most_common to sort most frequent glosses. Then writes each gloss (f.write) skipping a line. This file is sent to the mBART tokenizer
    for gloss, count in gloss_vocab.most_common():
        f.write(f"{gloss}\n")

# Saves statistics into a Key:Value dictionary. File is saved in json format (universal data interchange notation, identical to python dictionary in text)
stats = {
    'n_glosses_tokens': sum(gloss_vocab.values()),
    'unique_glosses_types': len(gloss_vocab),
    'train_size': len(splits['train']),
    'val_size': len(splits['val']),
    'test_size': len(splits['test'])
}
with open(os.path.join(VAR_PATH, 'stats.json'), 'w') as f:
    json.dump(stats, f)

print(f"Vocabulary saved. Total unique glosses: {len(gloss_vocab)}")
print("Process completed successfully!")