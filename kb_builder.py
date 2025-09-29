import argparse
from datsets import load_dataset
import pandas as pd
from utils import ( ensure_dir, load_emb, embed_texts, build_faiss_index, save_faiss_index, save_mapping)
import os

DEFAULT_DIR = 'kb_data'

def prepare_dataframe_from_dataset(dataset_name: str, split: str, max_rows: int= None):
    dataset= load_dataset(dataset_name, split= split)
    rows= []
    for i, row in enumerate(dataset):
        if max_rows and i >= max_rows:
            break
        q = None
        a= None
        if 'questions' in row:
            q = row['question']
        elif 'query' in row:
            q= row['query']
        elif 'title' in row and 'context' in row:
            q= row['title']
        if 'answers' in row and isinstance(row['answers'], dict):
            texts= row['answers'].get('text', [])
            a = texts[0] if texts else ""
            