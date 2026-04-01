import sys
import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from rdkit import Chem
from scipy import sparse as sp
from tqdm import tqdm
import warnings
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
sys.path.append("..")
try:
    from tools.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
except ImportError:
    print(
        "Error: Cannot import RDKit2DNormalized, please make sure the script is running in the correct project directory.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="KAMT Antibiotics Finetune Preprocessing")
    parser.add_argument("--base_dir", type=str, default="../dataset/antibiotics")
    parser.add_argument("--split_type", type=str, default='scaffold', choices=['random', 'scaffold'], required=True)
    parser.add_argument("--n_jobs", type=int, default=16)
    return parser.parse_args()


def extract_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512))
    except:
        return None


def extract_descriptor(smiles):
    generator = RDKit2DNormalized()
    try:
        res = generator.process(smiles)
        if res is None: return None
        data = res[1:] if isinstance(res[0], (str, bytes)) else res
        return data
    except:
        return None


def process_single_csv(csv_path, output_dir, n_jobs):
    print(f"\nProcessing file: {csv_path}")
    df = pd.read_csv(csv_path)
    smiles_list = df['smiles'].tolist()
    labels = df['label'].tolist()
    total = len(smiles_list)
    print(f"Extracting descriptor...")
    with Pool(n_jobs) as pool:
        desc_results = list(tqdm(pool.imap(extract_descriptor, smiles_list), total=total, desc="descriptor"))

    print(f"Extracting fingerprint...")
    with Pool(n_jobs) as pool:
        fp_results = list(tqdm(pool.imap(extract_fingerprint, smiles_list), total=total, desc="fingerprint"))

    valid_indices = []
    for i in range(total):
        if desc_results[i] is not None and fp_results[i] is not None:
            valid_indices.append(i)

    print(f"Alignment completed-Significant figures: {len(valid_indices)}/{total} ")

    final_df = df.iloc[valid_indices].copy()
    final_descs = np.array([desc_results[i] for i in valid_indices], dtype=np.float32)
    final_fps = np.array([fp_results[i] for i in valid_indices], dtype=np.int8)

    file_prefix = os.path.splitext(os.path.basename(csv_path))[0]

    np.savez_compressed(os.path.join(output_dir, f"{file_prefix}_desc.npz"), md=final_descs)
    sp.save_npz(os.path.join(output_dir, f"{file_prefix}_fp.npz"), sp.csc_matrix(final_fps))

    final_df.to_csv(os.path.join(output_dir, f"{file_prefix}_aligned.csv"), index=False)
    print("Saved descriptor and fingerprint!")


def main():
    args = parse_args()
    data_folder = os.path.join(args.base_dir, args.split_type)

    if not os.path.exists(data_folder):
        print(f"Directory does not exist: {data_folder}")
        return
    for split in ['train', 'val', 'test']:
        csv_file = os.path.join(data_folder, f"{split}.csv")
        if os.path.exists(csv_file):
            process_single_csv(csv_file, data_folder, args.n_jobs)
        else:
            print(f"Skip {split}.csv (file does not exist)")

    print("\nAll feature extraction tasks have been completed!")


if __name__ == '__main__':
    main()
