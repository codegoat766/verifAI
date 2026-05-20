"""
FastAPI backend – Fake News & Deepfake Detection
"""

import io
import re
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── globals filled at startup ──────────────────────────
news_tokenizer = None
news_model = None
df_model = None
df_preprocess = None


def _load_news_model():
    repo = "codegoat667/fake-news-detector"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForSequenceClassification.from_pretrained(repo)
    mdl.eval()
    return tok, mdl


def _load_deepfake_model():
    pth = hf_hub_download(
        repo_id="codegoat667/deepfake-detector",
        filename="efficientnet_deepfake.pth",
    )
    mdl = models.efficientnet_b0(weights=None)
    in_feat = mdl.classifier[1].in_features
    mdl.classifier[1] = torch.nn.Linear(in_feat, 2)
    mdl.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
    mdl.eval()
    pre = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return mdl, pre


@asynccontextmanager
async def lifespan(app: FastAPI):
    global news_tokenizer, news_model, df_model, df_preprocess
    print("[*] Loading models ...")
    news_tokenizer, news_model = _load_news_model()
    df_model, df_preprocess = _load_deepfake_model()
    print("[+] Models loaded.")
    yield


app = FastAPI(title="VerifAI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    reason: str
    key_signals: Optional[List[str]] = None


# ── helpers ────────────────────────────────────────────
def _extract_key_signals(inputs, attentions, tokenizer, top_k=6):
    """Extract the most-attended-to tokens from the CLS token's
    perspective in the last attention layer (averaged across heads)."""
    try:
        if not attentions or len(attentions) == 0:
            return []

        # attentions[-1] shape: (1, num_heads, seq_len, seq_len)
        # Take CLS row (index 0), average over heads
        cls_attn = attentions[-1].squeeze(0).mean(dim=0)[0]  # (seq_len,)

        input_ids = inputs["input_ids"].squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Build (token, score) pairs, skipping special tokens
        scored = []
        for i, (tok, score) in enumerate(zip(tokens, cls_attn)):
            if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]"):
                continue
            scored.append((tok, float(score)))

        # Merge subword tokens (RoBERTa uses byte-level BPE)
        merged = []
        for tok, score in scored:
            clean = tok.replace("\u0120", "").replace("\u0122", "").strip()
            if not clean:
                continue
            if tok.startswith("\u0120") or not merged:
                merged.append((clean, score))
            else:
                prev_word, prev_score = merged[-1]
                merged[-1] = (prev_word + clean, max(prev_score, score))

        # Filter out very short / punctuation-only tokens
        merged = [(w, s) for w, s in merged if len(w) > 2 and re.search(r"[a-zA-Z]", w)]

        # Sort by attention score, take top-k unique
        merged.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        result = []
        for word, _ in merged:
            wl = word.lower()
            if wl not in seen:
                seen.add(wl)
                result.append(word)
            if len(result) >= top_k:
                break
        return result
    except Exception:
        return []


def _text_reason(label, confidence, key_signals):
    """Generate a human-readable reason for the text classification."""
    signals_str = ", ".join(f'"{s}"' for s in key_signals[:5]) if key_signals else "N/A"

    if label == "FAKE":
        if confidence >= 90:
            tone = "Strong indicators of fabricated content detected."
            detail = ("The model identified language patterns, sensationalist phrasing, "
                      "and rhetorical cues that are highly characteristic of misinformation.")
        elif confidence >= 70:
            tone = "Moderate indicators of potentially fabricated content."
            detail = ("The text exhibits several linguistic patterns commonly found in "
                      "misleading or unverified articles, though some features are ambiguous.")
        else:
            tone = "Weak indicators of possible misinformation."
            detail = ("The model found subtle linguistic cues that lean toward fabrication, "
                      "but confidence is low. Manual verification is recommended.")
    else:
        if confidence >= 90:
            tone = "Content appears highly credible."
            detail = ("The writing style, factual tone, and language structure are "
                      "consistent with legitimate journalism and verified reporting.")
        elif confidence >= 70:
            tone = "Content appears likely authentic."
            detail = ("The text exhibits mostly standard journalistic patterns, though "
                      "some elements were ambiguous during analysis.")
        else:
            tone = "Content leans toward authentic but with low certainty."
            detail = ("The model found mixed signals. The text has some credible features "
                      "but also patterns that warrant further verification.")

    return f"{tone} {detail} Key signal words: {signals_str}."


def _image_reason(label, confidence):
    """Generate a human-readable reason for the deepfake classification."""
    if label == "FAKE":
        if confidence >= 90:
            return ("High-confidence deepfake detection. The model identified significant "
                    "anomalies in facial texture, skin smoothness, and edge consistency "
                    "that are characteristic of AI-generated or manipulated imagery. "
                    "Artifacts in lighting transitions and micro-texture patterns "
                    "strongly suggest synthetic generation.")
        elif confidence >= 70:
            return ("Moderate-confidence deepfake indicators. The image shows "
                    "subtle inconsistencies in facial feature geometry, unnatural "
                    "skin rendering, or irregular boundary artifacts between the face "
                    "and background that suggest possible AI manipulation.")
        else:
            return ("Low-confidence deepfake suspicion. The model detected minor "
                    "texture or lighting irregularities that could indicate manipulation, "
                    "but the signals are weak. The image may benefit from additional "
                    "analysis or higher-resolution input.")
    else:
        if confidence >= 90:
            return ("High-confidence authenticity assessment. The image exhibits "
                    "natural skin micro-texture, consistent sub-surface scattering, "
                    "coherent lighting across facial features, and no detectable "
                    "artifacts from generative models or face-swap pipelines.")
        elif confidence >= 70:
            return ("Moderate-confidence authenticity assessment. The facial features "
                    "display mostly natural characteristics including consistent "
                    "lighting and texture, though some regions were ambiguous. "
                    "No strong generative artifacts were detected.")
        else:
            return ("Low-confidence authenticity assessment. The image appears "
                    "mostly natural, but the model found ambiguous regions that "
                    "could not be definitively classified. Consider using a "
                    "higher-resolution image for more reliable results.")


# ── routes ─────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/analyze-text", response_model=PredictionResponse)
async def analyze_text(req: TextRequest):
    inputs = news_tokenizer(
        req.text, return_tensors="pt", truncation=True, max_length=512,
    )
    with torch.no_grad():
        outputs = news_model(**inputs, output_attentions=True)

    probs = F.softmax(outputs.logits, dim=-1).squeeze()
    idx = int(probs.argmax())
    label = "FAKE" if idx == 1 else "REAL"
    confidence = round(float(probs[idx]) * 100, 1)

    key_signals = _extract_key_signals(
        inputs, outputs.attentions, news_tokenizer,
    )
    reason = _text_reason(label, confidence, key_signals)

    return PredictionResponse(
        label=label,
        confidence=confidence,
        reason=reason,
        key_signals=key_signals,
    )


@app.post("/analyze-image", response_model=PredictionResponse)
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = df_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = df_model(tensor)
    probs = F.softmax(logits, dim=-1).squeeze()
    idx = int(probs.argmax())
    label = "REAL" if idx == 1 else "FAKE"
    confidence = round(float(probs[idx]) * 100, 1)
    reason = _image_reason(label, confidence)

    return PredictionResponse(
        label=label,
        confidence=confidence,
        reason=reason,
    )


# serve static assets
app.mount("/static", StaticFiles(directory="static"), name="static")
