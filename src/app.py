"""
University Intent Classification — Streamlit App
Run with:  streamlit run src/app.py
"""

import os
import json
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ── Office info (icon + description shown after prediction) ───────────────
OFFICE_INFO = {
    "Academic Office": {
        "icon": "🎓",
        "desc": "Handles course registration, grades, transcripts, and academic records.",
        "location": "Building A, Room 101",
    },
    "Finance Office": {
        "icon": "💰",
        "desc": "Handles tuition fees, scholarships, payments, and financial aid.",
        "location": "Building B, Room 202",
    },
    "IT Support": {
        "icon": "💻",
        "desc": "Handles student email, campus Wi-Fi, LMS access, and device issues.",
        "location": "Building C, Room 305 — or email it@university.edu",
    },
    "Administration": {
        "icon": "🏛️",
        "desc": "Handles general university policies, official letters, and campus services.",
        "location": "Main Building, Room 001",
    },
    "Hostel Office": {
        "icon": "🏠",
        "desc": "Handles dormitory applications, room assignments, and hostel complaints.",
        "location": "Student Housing Block, Ground Floor",
    },
}

# ── Load model (cached so it only loads once) ─────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        return None, None, None

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    with open(os.path.join(MODEL_DIR, 'label_map.json')) as f:
        label_map = json.load(f)

    return model, tokenizer, label_map


# ── Inference ─────────────────────────────────────────────────────────────
def predict(text: str, model, tokenizer, label_map, max_len: int = 128):
    enc = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    with torch.no_grad():
        outputs = model(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
        )
    probs   = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = probs.argmax().item()
    label   = label_map[str(pred_id)]
    confidence = float(probs[pred_id])

    # Top-3 probabilities for the bar chart
    top3_idx    = probs.topk(min(3, len(label_map))).indices.tolist()
    top3_labels = [label_map[str(i)] for i in top3_idx]
    top3_probs  = [float(probs[i]) for i in top3_idx]

    return label, confidence, top3_labels, top3_probs


# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="University Help Desk",
    page_icon="🏫",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────
st.title("🏫 University Help Desk")
st.markdown(
    "Describe your question or problem below, "
    "and we'll direct you to the right office."
)
st.divider()

# ── Load model ────────────────────────────────────────────────────────────
model, tokenizer, label_map = load_model()

if model is None:
    st.error(
        "⚠️ Model not found. "
        "Please run **model.ipynb** first to train and save the model, "
        "then restart this app."
    )
    st.stop()

# ── Input ─────────────────────────────────────────────────────────────────
query = st.text_area(
    "Your question:",
    placeholder="e.g. I haven't received my scholarship payment this month...",
    height=120,
)

col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Submit", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.rerun()

# ── Result ────────────────────────────────────────────────────────────────
if submitted:
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Analyzing your query…"):
            label, confidence, top3_labels, top3_probs = predict(
                query, model, tokenizer, label_map
            )

        st.divider()

        # ── Main result card ──────────────────────────────────────────────
        info = OFFICE_INFO.get(label, {"icon": "🏢", "desc": "", "location": ""})

        st.success(f"### {info['icon']} Please go to: **{label}**")

        col_conf, col_loc = st.columns(2)
        with col_conf:
            st.metric("Confidence", f"{confidence:.0%}")
        with col_loc:
            if info.get("location"):
                st.info(f"📍 {info['location']}")

        if info.get("desc"):
            st.markdown(f"_{info['desc']}_")

        # ── Top-3 probability bar ─────────────────────────────────────────
        with st.expander("See probability breakdown"):
            for lbl, prob in zip(top3_labels, top3_probs):
                ico = OFFICE_INFO.get(lbl, {}).get("icon", "🏢")
                st.write(f"{ico} **{lbl}**")
                st.progress(prob, text=f"{prob:.1%}")

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption("University Help Desk — NLP Intent Classification Project")