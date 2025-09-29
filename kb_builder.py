import argparse
from datasets import load_dataset
import pandas as pd
from utils import (ensure_dir, load_emb, embed_texts, build_faiss_index,
                   save_faiss_index, save_mapping, save_pickle)
import os

DEFAULT_OUT_DIR = "kb_data"

def prepare_dataframe_from_dataset(dataset_name: str, split: str, max_rows: int = None) -> pd.DataFrame:
    dataset = load_dataset(dataset_name, split=split)
    rows = []
    for i, row in enumerate(dataset):
        if max_rows and i >= max_rows:
            break
        q = None
        a = None
        if 'question' in row:
            q = row['question']
        elif 'query' in row:
            q = row['query']
        elif 'title' in row and 'context' in row:
            q = row['title']
        if 'answers' in row and isinstance(row['answers'], dict):
            texts = row['answers'].get('text', [])
            a = texts[0] if texts else ""
        elif 'answer' in row and isinstance(row['answer'], str):
            a = row['answer']
        elif 'context' in row:
            a = row['context']
        else:
            a = ""
        if not q:
            continue
        rows.append({"question": str(q), "amswer": str(a)})
    df = pd.DataFrame(rows)
    return df

def main(args):
    outdir = args.outdir
    ensure_dir(outdir)
    print(f"Loading datset '{args.dataset}' split '{args.split} ...")
    df = prepare_dataframe_from_dataset(args.dataset, args.split, args.max_rows)
    print(f"Prepared dataframe with {len(df)} rows.")

    mapping_path = os.path.join(outdir, "kb_mapping.csv")
    save_mapping(df, mapping_path)
    print(f"Saved mapping to {mapping_path}")

    embedder = load_emb(args.embed_model)
    texts = df["question"].tolist()
    print("Computing embedding (questions) ...")
    embedding = embed_texts(embedder, texts, batch_size=args.batch_size)
    print("Embedding computed.shape:", embedding.shape)

    print("Building FAISS index ...")
    index = build_faiss_index(embedding)
    idx_path = os.path.join(outdir, "kb.index")
    save_faiss_index(index, idx_path)
    print(f"FAISS index saved to {idx_path}")
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,default="squad", help="HF datset")
    parser.add_argument("--split", type=str, default="train", help="Data split")
    parser.add_argument("--max_rows", type=int, default=50000, help="Max rows to")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUT_DIR, help="output")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2",help="embed model")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()
    main(args)