# app.py
import streamlit as st
import os
import numpy as np
import faiss
from utils import (
    ensure_dir, load_emb, load_faiss_index, load_mapping,
    tokk_search
)
from utils import EMBED_MODEL_NAME
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import textwrap

KB_DIR = "kb_data"
INDEX_PATH = os.path.join(KB_DIR, "kb.index")
MAPPING_PATH = os.path.join(KB_DIR, "kb_mapping.csv")

# Config defaults
DEFAULT_TOPK = 5
GEN_MODEL = "google/flan-t5-base"  # change if you want a different generator
MAX_CONTEXT_CHARS = 4000  # keep total context length in check

@st.cache_resource
def load_resources(embed_model=EMBED_MODEL_NAME, gen_model_name=GEN_MODEL):
    # Embedding model
    embedder = load_emb(embed_model)

    # Load mapping and index
    if not os.path.exists(MAPPING_PATH) or not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Missing KB files. Run kb_builder.py to create {MAPPING_PATH} and {INDEX_PATH}")

    df = load_mapping(MAPPING_PATH)
    index = faiss.read_index(INDEX_PATH)

    # Generation pipeline
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    if device == 0:
        model = model.to("cuda")
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        framework="pt"
    )
    return embedder, df, index, gen_pipeline

def retrieve(embedder, index, df, query: str, topk: int = DEFAULT_TOPK):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    # normalize
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb.astype(np.float32)
    D, I = index.search(q_emb, topk)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        results.append({"question": row["question"], "answer": row["answer"]})
    return results, D[0]

def build_context(kb_results, max_chars=MAX_CONTEXT_CHARS):
    """Combine retrieved answers into a single context string (trim long ones)."""
    parts = []
    total = 0
    for r in kb_results:
        piece = f"Q: {r['question']}\nA: {r['answer']}\n"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n".join(parts)

def build_prompt(user_question: str, context: str):
    prompt = textwrap.dedent(f"""
    Use the provided context to answer the user's question as helpfully and concisely as possible.
    Context:
    {context}

    User question: {user_question}

    Provide a clear, direct answer. If the context doesn't contain enough info, say you don't know and optionally provide next steps.
    """)
    return prompt.strip()

def generate_answer(gen_pipeline, prompt: str, max_length: int = 256, temperature: float = 0.0):
    # generation params - adjust as needed
    out = gen_pipeline(prompt, max_length=max_length, do_sample=temperature > 0.0, temperature=temperature)
    return out[0]["generated_text"].strip()

# --- Streamlit app UI ---
st.set_page_config(page_title="AI-Powered Chat Assistant (HF)", layout="wide")
st.title("💬 AI-Powered Chat Assistant!!")

with st.sidebar:
    st.header("Settings")
    dataset_info = st.text_input("KB data directory", KB_DIR)
    topk = st.slider("Top-k retrieval", min_value=1, max_value=20, value=5)
    gen_max_len = st.slider("Generation max length", min_value=50, max_value=512, value=256)
    gen_temperature = st.slider("Generation temperature", min_value=0.0, max_value=1.0, value=0.0)
    if st.button("Reload resources"):
        st.cache_resource.clear()

# Load resources (cached)
try:
    embedder, df, index, gen_pipeline = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, assistant)

col1, col2 = st.columns([2, 3])
with col1:
    user_question = st.text_input("Ask a question:", key="input")
    if st.button("Send"):
        if not user_question or user_question.strip() == "":
            st.warning("Please type a question.")
        else:
            # Retrieve
            kb_results, scores = retrieve(embedder, index, df, user_question, topk=topk)
            context = build_context(kb_results)
            prompt = build_prompt(user_question, context)
            # Generate
            try:
                answer = generate_answer(gen_pipeline, prompt, max_length=gen_max_len, temperature=gen_temperature)
            except Exception as ex:
                st.error(f"Generation failed: {ex}")
                answer = "Sorry — generation failed."

            # Save to history
            st.session_state.history.append({
                "user": user_question,
                "answer": answer,
                "kb": kb_results
            })

with col2:
    st.subheader("Conversation")
    for item in reversed(st.session_state.history):
        st.markdown(f"**User:** {item['user']}")
        st.markdown(f"**Assistant:** {item['answer']}")
        with st.expander("Show retrieved KB results"):
            for r in item["kb"]:
                st.markdown(f"- **Q:** {r['question']}")
                st.markdown(f"  - **A:** {r['answer']}")

st.markdown("---")
st.markdown("**KB stats**")
st.write(f"Indexed rows: {len(df)}")
st.write(df.head(5))
