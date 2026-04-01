import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def standardize_smiles(smi, remover):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        mol = remover.StripMol(mol)
        smi_list = Chem.MolToSmiles(mol).split('.')
        if len(smi_list) > 1:
            smi = max(smi_list, key=len)
            mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except:
        return None


def generate_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return "None"
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold_mol)
    except:
        return "None"


def scaffold_split(df, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    scaffolds = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="计算骨架"):
        scaffold = generate_scaffold(row['smiles'])
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx)
    scaffold_list = sorted(scaffolds.values(), key=len, reverse=True)

    train_indices, val_indices, test_indices = [], [], []
    n_train = int(len(df) * train_size)
    n_val = int(len(df) * val_size)

    for group in scaffold_list:
        if len(train_indices) + len(group) <= n_train:
            train_indices.extend(group)
        elif len(train_indices) + len(val_indices) + len(group) <= n_train + n_val:
            val_indices.extend(group)
        else:
            test_indices.extend(group)
    return df.loc[train_indices], df.loc[val_indices], df.loc[test_indices]


def save_datasets(train, val, test, path):
    os.makedirs(path, exist_ok=True)
    train.to_csv(os.path.join(path, "train.csv"), index=False)
    val.to_csv(os.path.join(path, "val.csv"), index=False)
    test.to_csv(os.path.join(path, "test.csv"), index=False)
    print(f"Saved at: {path}")


def main():
    input_file = "../dataset/antibiotics/raw.csv"
    if not os.path.exists(input_file):
        print(f"Input file not found {input_file}")
        return
    df = pd.read_csv(input_file)
    print(f"Original data volume: {len(df)}")
    remover = SaltRemover()
    tqdm.pandas(desc="Standardized processing")
    df['smiles'] = df['smiles'].progress_apply(lambda x: standardize_smiles(x, remover))
    df = df.dropna(subset=['smiles']).drop_duplicates(subset=['smiles'])
    print(f"Data volume after cleaning: {len(df)}")
    print("Random splitting in progress...")
    train_rand, temp_rand = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_rand, test_rand = train_test_split(temp_rand, test_size=0.5, random_state=42, stratify=temp_rand['label'])
    save_datasets(train_rand, val_rand, test_rand, "../dataset/antibiotics/random/")
    print(f"Train: {len(train_rand)} | Val: {len(val_rand)} | Test: {len(test_rand)}")
    print("Scaffold splitting in progress...")
    train_scaf, val_scaf, test_scaf = scaffold_split(df)
    save_datasets(train_scaf, val_scaf, test_scaf, "../dataset/antibiotics/scaffold/")
    print(f"Train: {len(train_scaf)} | Val: {len(val_scaf)} | Test: {len(test_scaf)}")
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()
