"""
Fake News & Deepfake Detection Web App
──────────────────────────────────────
Tab 1 – Paste article text  → REAL / FAKE + confidence
Tab 2 – Upload face image   → REAL / FAKE + confidence

Models hosted on HuggingFace:
  • codegoat667/fake-news-detector   (DistilRoBERTa)
  • codegoat667/deepfake-detector    (EfficientNet-B0 state-dict)
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
import os

# ───────────────────── page config ─────────────────────
st.set_page_config(
    page_title="VerifAI – Fake News & Deepfake Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ───────────────────── custom CSS ──────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── globals ── */
    :root {
        --bg-primary: #0a0e17;
        --bg-card: #111827;
        --bg-card-hover: #1a2332;
        --accent-cyan: #06d6a0;
        --accent-pink: #ef476f;
        --accent-blue: #118ab2;
        --accent-yellow: #ffd166;
        --text-primary: #f0f4f8;
        --text-muted: #8899aa;
        --radius: 16px;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* ── header banner ── */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #06d6a0, #118ab2, #ef476f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: .15rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: var(--text-muted);
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* ── cards ── */
    .result-card {
        background: linear-gradient(145deg, #111827 0%, #1a2332 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 2rem 2.2rem;
        margin-top: 1.4rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        animation: fadeUp 0.5s ease-out;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .verdict {
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: .3rem;
    }
    .verdict.real { color: #06d6a0; }
    .verdict.fake { color: #ef476f; }
    .confidence-label {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.92rem;
        margin-bottom: .6rem;
    }

    /* ── progress bar ── */
    .bar-track {
        width: 100%;
        height: 14px;
        background: rgba(255,255,255,0.07);
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: .5rem;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s cubic-bezier(.22,1,.36,1);
    }
    .bar-fill.real { background: linear-gradient(90deg, #06d6a0, #118ab2); }
    .bar-fill.fake { background: linear-gradient(90deg, #ef476f, #ffd166); }

    .pct {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .pct.real { color: #06d6a0; }
    .pct.fake { color: #ef476f; }

    /* ── tab styling ── */
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1rem;
    }

    /* ── footer ── */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-bottom: 1.5rem;
        opacity: 0.6;
    }

    /* hide default streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────── hero ────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ VerifAI</h1>
        <p>Detect fake news articles &amp; AI-generated face images in seconds</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ───────────────────── model loaders (cached) ──────────
@st.cache_resource(show_spinner="Loading fake-news model …")
def load_news_model():
    """Load DistilRoBERTa fine-tuned for binary news classification."""
    model_name = "codegoat667/fake-news-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading deepfake model …")
def load_deepfake_model():
    """Load EfficientNet-B0 with custom head from HuggingFace state-dict."""
    # Download the .pth file from the Hub
    pth_path = hf_hub_download(
        repo_id="codegoat667/deepfake-detector",
        filename="efficientnet_deepfake.pth",
    )
    # Build the architecture – match training config (2 classes)
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 2)
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    # Standard ImageNet preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess


# ───────────────────── result card helper ──────────────
def render_result(label: str, confidence: float):
    """Render a styled verdict card with animated progress bar."""
    css_class = "real" if label == "REAL" else "fake"
    icon = "✅" if label == "REAL" else "🚨"
    pct = confidence * 100
    st.markdown(
        f"""
        <div class="result-card">
            <div class="verdict {css_class}">{icon} {label}</div>
            <div class="confidence-label">Confidence</div>
            <div class="bar-track">
                <div class="bar-fill {css_class}" style="width:{pct:.1f}%"></div>
            </div>
            <div class="pct {css_class}">{pct:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────── tabs ────────────────────────────
tab_news, tab_image = st.tabs(["📰  News Article", "🖼️  Face Image"])

# ── Tab 1: Fake News ──────────────────────────────────
with tab_news:
    st.markdown("#### Paste an article or headline below")
    article_text = st.text_area(
        "Article text",
        height=200,
        placeholder="Paste the full article or headline here …",
        label_visibility="collapsed",
    )

    if st.button("🔍  Analyse Article", use_container_width=True, key="btn_news"):
        if not article_text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Analysing text …"):
                tokenizer, model = load_news_model()
                inputs = tokenizer(
                    article_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1).squeeze()
                # 0 = REAL, 1 = FAKE
                pred_idx = int(probs.argmax())
                label = "FAKE" if pred_idx == 1 else "REAL"
                confidence = float(probs[pred_idx])
            render_result(label, confidence)

# ── Tab 2: Deepfake ───────────────────────────────────
with tab_image:
    st.markdown("#### Upload a face image to check for AI manipulation")
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("🔍  Analyse Image", use_container_width=True, key="btn_img"):
        if uploaded is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Analysing image …"):
                model, preprocess = load_deepfake_model()
                tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tensor)
                probs = F.softmax(logits, dim=-1).squeeze()
                # 0 = fake, 1 = real
                pred_idx = int(probs.argmax())
                label = "REAL" if pred_idx == 1 else "FAKE"
                confidence = float(probs[pred_idx])
            render_result(label, confidence)

# ───────────────────── footer ──────────────────────────
st.markdown(
    '<div class="footer">VerifAI · Models by codegoat667 · Powered by Streamlit & PyTorch</div>',
    unsafe_allow_html=True,
)
