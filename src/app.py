import os
import json
import random
import torch
import joblib
import streamlit as st
import re
from transformers import BertTokenizer, BertForSequenceClassification

# ── Text Normalization ─────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\bim\b", "i'm", text)
    text = re.sub(r"\bive\b", "i've", text)
    text = re.sub(r"\bid\b", "i'd", text)
    text = re.sub(r"\bcant\b", "can't", text)
    text = re.sub(r"\bdont\b", "don't", text)
    text = re.sub(r"\bwont\b", "won't", text)
    text = re.sub(r"\bisnt\b", "isn't", text)
    text = re.sub(r"\barent\b", "aren't", text)
    return text

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))          # now = src/
# Pointing exactly to your safe, un-committed local BERT directory
MODEL_DIR    = os.path.join(BASE_DIR, 'saved_bert_base')           # src/saved_bert_base ✅
# Path to your JSON file to retrieve the answers/responses
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'intents.json')  # src/../dataset/intents.json ✅

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Intent Classification Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ── Load Model & Data (Cached) ────────────────────────────────────────────
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_DIR):
        return None, None, None, None
        
    # 1. Load native transformer files
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    # 2. Load the Label Encoder saved by joblib
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)
    else:
        return None, None, None, None

    # 3. Load the dataset responses
    responses_dict = {}
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'r') as f:
            data = json.load(f)
            # Adjust mapping key depending on if your JSON uses an "intents" list
            intents_list = data.get('intents', data) if isinstance(data, dict) else data
            for item in intents_list:
                # Store responses grouped by their intent text label
                responses_dict[item['tag']] = item.get('responses', ["I understand. Please tell me more."])
                
    return model, tokenizer, label_encoder, responses_dict

# ── Inference ─────────────────────────────────────────────────────────────
def predict_intent(text: str, model, tokenizer, label_encoder, max_len: int = 256):
    enc = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Detect execution device automatically (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=enc['input_ids'].to(device),
            attention_mask=enc['attention_mask'].to(device)
        )
        
    probs   = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = probs.argmax().item()
    
    # Decode integer index back to text string label using the Label Encoder
    intent_label = label_encoder.inverse_transform([pred_id])[0]
    confidence = float(probs[pred_id])
    
    return intent_label, confidence

# ── Initialization ────────────────────────────────────────────────────────
model, tokenizer, label_encoder, responses_dict = load_resources()

if model is None:
    st.error(
        f"⚠️ Resources not found.\n\n"
        f"Make sure your trained model files sit in: `{MODEL_DIR}`\n"
        f"And your dataset sits in: `{DATASET_PATH}`"
    )
    st.stop()

# Initialize session state for persistent chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I'm a specialized assistant trained to classify mental health concerns. How are you feeling today?"}
    ]

# ── Header UI ─────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 90% !important;        /* ← add this */
        padding-left: 15rem !important;    /* ← and this */
        padding-right: 15rem !important;   /* ← and this */
    }
    /* Reduce gap between title and the columns below */
    .block-container h1 {
        margin-bottom: 0rem !important;
    }
    /* Tighten spacing between all stacked elements */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0rem !important;
    }
    /* Divider spacing */
    hr {
        margin: 0.3rem 0 !important;
    }
    /* Add breathing room above the Clear Chat button */
    div[data-testid="stVerticalBlock"] > div:has(button) {
        padding-top: 0.75rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Main Application Branding
st.title("Mental Health Intent Classification Chatbot")

# Shrink the metric value font size
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1rem !important; }
    </style>
""", unsafe_allow_html=True)

# 2. Split Meta Info & Metrics into clean, horizontal columns
meta_col, metric_col = st.columns([3, 2], gap="medium")

with meta_col:
    st.markdown("**Authors:**")
    st.caption("• **Danniel Prananda** - 0706012310034")
    st.caption("• **Kevin Artan** - 0706012310032")

with metric_col:
    st.markdown("### Model Framework")
    st.metric(
        label="Fine-Tuned BERT-base-uncased", 
        value="94.0% Accuracy", 
        help="Evaluated against strict test benchmarks across 9 mental health dialog intents."
    )

st.divider()

# Clear Chat Button
if st.button("Clear Chat History", type="secondary"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I'm a specialized assistant trained to classify mental health concerns. How are you feeling today?"}
    ]
    st.rerun()


# ── Render Chat History ───────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── User Interaction Loop ─────────────────────────────────────────────────
if user_input := st.chat_input("Type your message here..."):

    # normalize text
    normalized_input = normalize_text(user_input)
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": normalized_input})
    with st.chat_message("user"):
        st.markdown(normalized_input)
        
    # 2. Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Processing intent..."):
            # Run prediction
            intent, confidence = predict_intent(normalized_input, model, tokenizer, label_encoder)
            
            # Fetch response based on predicted intent
            # available_responses = responses_dict.get(intent, ["I hear you. Could you clarify that for me?"])
            # bot_response = random.choice(available_responses)
            CONFIDENCE_THRESHOLD = 0.4

            if confidence < CONFIDENCE_THRESHOLD:
                bot_response = "I'm not quite sure I understood that. Could you rephrase or tell me more about how you're feeling?"
            else:
                available_responses = responses_dict.get(intent, ["I hear you. Could you clarify that for me?"])
                bot_response = random.choice(available_responses)
            
        # Display the text response
        st.markdown(bot_response)
        
        # Display diagnostic meta-information in a subtle expander block
        with st.expander("🛠️ Model Diagnosis"):
            st.write(f"**Predicted Intent:** `{intent}`")
            st.write(f"**Classification Confidence:** `{confidence:.2%}`")
            
    # Save Assistant Response to History
    st.session_state.messages.append({"role": "assistant", "content": bot_response})