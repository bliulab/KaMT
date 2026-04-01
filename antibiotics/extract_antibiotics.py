import sqlite3
import pandas as pd
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(ROOT_DIR, 'dataset', 'chembl', 'chembl_36.db')
OUTPUT_RAW_CSV = os.path.join(ROOT_DIR, 'dataset', 'antibiotics', 'raw.csv')

def extract_antibiotics_data():
    if not os.path.exists(DB_PATH):
        print(f"Database file not found, please check the path: {DB_PATH}")
        return
    conn = sqlite3.connect(DB_PATH)
    pos_query = """
    SELECT DISTINCT CS.CANONICAL_SMILES as smiles, 1 as label
    FROM COMPOUND_STRUCTURES CS
    JOIN MOLECULE_ATC_CLASSIFICATION MATC ON CS.MOLREGNO = MATC.MOLREGNO
    WHERE MATC.LEVEL5 LIKE 'J01%' AND CS.CANONICAL_SMILES IS NOT NULL
    """
    try:
        print("Extracting positive antibiotic samples from ChEMBL...")
        pos_df = pd.read_sql(pos_query, conn)
        n_pos = len(pos_df)
        if n_pos == 0:
            print("No positive samples found! Please check whether the database table MOLECULE_ATC_CLASSIFICATION has data.")
            return
        print(f"Found positive samples: {n_pos} ")
        n_neg_target = n_pos * 5
        print(f"Extracting non-antibiotic negative samples(target quantity: {n_neg_target})...")
        neg_query = f"""
        SELECT DISTINCT CS.CANONICAL_SMILES as smiles, 0 as label
        FROM COMPOUND_STRUCTURES CS
        WHERE CS.CANONICAL_SMILES IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM MOLECULE_ATC_CLASSIFICATION MATC 
            WHERE MATC.MOLREGNO = CS.MOLREGNO 
            AND MATC.LEVEL5 LIKE 'J01%'
        )
        ORDER BY RANDOM() LIMIT {n_neg_target}
        """
        neg_df = pd.read_sql(neg_query, conn)
        print(f"Extract negative samples: {len(neg_df)} ")
        df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42)
        df = df.drop_duplicates(subset=['smiles'])
        os.makedirs(os.path.dirname(OUTPUT_RAW_CSV), exist_ok=True)
        df.to_csv(OUTPUT_RAW_CSV, index=False)
        print("-" * 30)
        print(f"Task completed! The dataset has been successfully saved.")
        print(f"Final sample statistics - Positive samples: {len(df[df.label == 1])}, Negative samples: {len(df[df.label == 0])}")
        print(f"Save path: {OUTPUT_RAW_CSV}")
    except Exception as e:
        print(f"Extraction failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    extract_antibiotics_data()