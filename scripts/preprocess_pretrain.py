import sys
import os
import argparse
import numpy as np
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
    class RDKit2DNormalized:
        def process(self, s): return [s] + [0.0] * 717


def parse_args():
    parser = argparse.ArgumentParser(description="KAMT Data Preprocessing (Aligned Version)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--smi_filename", type=str, default="smiles.smi")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--extract_fp", action="store_true", default=True)
    parser.add_argument("--extract_desc", action="store_true", default=True)

    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.data_path
    if not args.extract_fp and not args.extract_desc:
        args.extract_fp = args.extract_desc = True
    return args


def extract_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512))
    except:
        return None


def extract_descriptor_safe(smiles):
    generator = RDKit2DNormalized()
    try:
        res = generator.process(smiles)
        if res is None: return None
        data = res[1:] if isinstance(res[0], str) else res
        return data
    except:
        return None


def preprocess_kamt_dataset(args):
    smi_file = os.path.join(args.data_path, args.smi_filename)
    if not os.path.exists(smi_file):
        print(f"Error: File not found {smi_file}")
        return

    print(f"Loading SMILES data: {args.smi_filename}")
    with open(smi_file, 'r') as f:
        smiless = [line.strip().split()[0] for line in f if line.strip()]

    total_count = len(smiless)
    fps_results = [None] * total_count
    desc_results = [None] * total_count

    if args.extract_fp:
        print(f"Task: Extracting RDKit Fingerprints...")
        with Pool(args.n_jobs) as pool:
            fps_results = list(
                tqdm(pool.imap(extract_fingerprint, smiless, chunksize=100), total=total_count, desc="Fingerprints"))

    if args.extract_desc:
        print(f"Task: Extracting Normalized Descriptors...")
        with Pool(args.n_jobs) as pool:
            desc_results = list(
                tqdm(pool.imap(extract_descriptor_safe, smiless, chunksize=50), total=total_count, desc="Descriptors"))

    print(f"Performing multi-modal alignment filtering...")
    valid_indices = []
    for i in range(total_count):
        fp_ok = (fps_results[i] is not None) if args.extract_fp else True
        desc_ok = (desc_results[i] is not None) if args.extract_desc else True

        if fp_ok and desc_ok:
            valid_indices.append(i)

    final_smiless = [smiless[i] for i in valid_indices]
    print(f"Alignment successful! Number of valid molecules: {len(final_smiless)} / Original total: {total_count}")

    if args.extract_fp:
        valid_fps = [fps_results[i] for i in valid_indices]
        fp_arr = np.array(valid_fps, dtype=np.int8)
        fp_sp_mat = sp.csc_matrix(fp_arr)
        fp_out = os.path.join(args.output_path, "kamt_pretrain_fps_512.npz")
        sp.save_npz(fp_out, fp_sp_mat)
        print(f"💾 Fingerprint saved (Shape: {fp_arr.shape})")

    if args.extract_desc:
        valid_descs = [desc_results[i] for i in valid_indices]
        md_data = np.array(valid_descs, dtype=np.float32)
        md_out = os.path.join(args.output_path, "kamt_knowledge_descriptors.npz")
        np.savez_compressed(md_out, md=md_data)
        print(f"💾 Descriptor saved (Shape: {md_data.shape})")

    cleaned_smi_name = "smiles_cleaned.smi"
    cleaned_smi_path = os.path.join(args.output_path, cleaned_smi_name)
    with open(cleaned_smi_path, 'w') as f:
        for s in final_smiless:
            f.write(f"{s}\n")
    print(f"💾 The aligned SMILES has been saved: {cleaned_smi_path}")


if __name__ == '__main__':
    args = parse_args()
    preprocess_kamt_dataset(args)
