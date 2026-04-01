import argparse
import sqlite3
from tqdm import tqdm


def extract_smiles_from_chembl(db_path, output_file):
    print(f"Connecting to the database: {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("Counting the total number of molecules...")
        cursor.execute("SELECT COUNT(*) FROM compound_structures WHERE canonical_smiles IS NOT NULL")
        total_count = cursor.fetchone()[0]
        print(f"Number of valid SMILES found: {total_count}")
        query = "SELECT canonical_smiles FROM compound_structures WHERE canonical_smiles IS NOT NULL"
        cursor.execute(query)
        print(f"Writing to file: {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in tqdm(cursor, total=total_count, desc="提取中"):
                smiles = row[0]
                if smiles:
                    f.write(f"{smiles}\n")
        print(f"Extraction complete! The file has been saved: {output_file}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--smiles_path", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    DB_NAME = args.data_path
    OUTPUT_NAME = args.smiles_path
    extract_smiles_from_chembl(DB_NAME, OUTPUT_NAME)
