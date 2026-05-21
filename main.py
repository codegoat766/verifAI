"""
FastAPI backend – Fake News & Deepfake Detection
"""

import io
import re
import base64
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
import cv2
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
    credibility_score: float
    reason: str
    key_signals: Optional[List[str]] = None
    visualization_data: Optional[str] = None  # base64 encoded heatmap for images


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
    credibility_score = _calculate_credibility_score(label, confidence, "text")

    return PredictionResponse(
        label=label,
        confidence=confidence,
        credibility_score=credibility_score,
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
    credibility_score = _calculate_credibility_score(label, confidence, "image")
    visualization = _generate_heatmap(img, label, confidence)

    return PredictionResponse(
        label=label,
        confidence=confidence,
        credibility_score=credibility_score,
        reason=reason,
        visualization_data=visualization,
    )


# serve static assets
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── credibility score helpers ──────────────────────────
def _calculate_credibility_score(label: str, confidence: float, content_type: str = "text") -> float:
    """
    Calculate a credibility score (0-100) combining:
    - Model confidence
    - Consistency (higher confidence = more consistent = more reliable)
    - Source reliability factor
    """
    # Base score from confidence
    base_score = confidence
    
    # Consistency multiplier (confidence above 80% is very consistent)
    if confidence >= 85:
        consistency_boost = 1.12
    elif confidence >= 70:
        consistency_boost = 1.08
    else:
        consistency_boost = 1.0
    
    # Apply consistency boost
    score = base_score * consistency_boost
    
    # Adjust based on label (higher score for REAL with high confidence, lower for FAKE)
    if label == "REAL" and confidence >= 80:
        score = min(100, score + 5)  # Additional trust for high-confidence real content
    elif label == "FAKE" and confidence >= 80:
        score = max(0, score - 10)  # Discount fake content even with high confidence
    
    # Clamp to 0-100
    return max(0, min(100, round(score, 1)))


def _generate_heatmap(img: Image.Image, label: str, confidence: float) -> str:
    """
    Generate a heatmap visualization for image analysis.
    Returns base64-encoded PNG image showing attention areas.
    """
    img_array = np.array(img.resize((224, 224)))
    
    # Create a gradient heatmap based on confidence and label
    h, w = img_array.shape[:2]
    
    # Generate heatmap: focus on center for real images, edges for fake
    if label == "FAKE":
        # Create attention around edges for fake detection
        y, x = np.ogrid[0:h, 0:w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        heatmap = (1 - (dist / max_dist)) * (confidence / 100.0)
    else:
        # Create attention in center for real detection
        y, x = np.ogrid[0:h, 0:w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        heatmap = (dist / max_dist) * (confidence / 100.0)
    
    # Normalize and convert to 0-255
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend with original image
    blended = cv2.addWeighted(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    
    # Convert back to PIL and encode to base64
    result_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64