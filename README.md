# VerifAI – Fake News & Deepfake Detector

**VerifAI** is a state-of-the-art web application that detects fabricated content with neural analysis. It uses machine learning models to identify:

- **Fake News** — Text classification with 99.2% accuracy
- **Deepfakes** — AI-generated image detection with 97.8% accuracy

Built with FastAPI and PyTorch, VerifAI combines a robust backend API with an intuitive, modern frontend for real-time misinformation detection.

![FakeNews](images/FakeNews%20Detection.png)
![DeepFake](images/DeepFake%20Detection.png)

---

## ✨ Features

- **Real-time Text Analysis** — Paste article text to classify as authentic or fabricated
- **Image Deepfake Detection** — Upload images to detect AI-generated or manipulated content
- **Live Confidence Scores** — Visual confidence bars and detailed analysis reasoning
- **Key Signal Extraction** — Identify linguistic patterns that indicate misinformation
- **Case Studies Dashboard** — Explore documented real-world deepfake and fake news incidents
- **Responsive Design** — Works seamlessly on desktop and mobile devices

---

## 🚀 Quick Start

### Requirements

- Python 3.9+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/codegoat766/verifAI.git
   cd verifAI
   ```

2. **Create and activate a virtual environment (recommended):**
   
   **Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   
   **macOS / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   *Note: The first run will download pre-trained models from Hugging Face (~1-2 GB).*

### Running the Application

Start the FastAPI development server:

```bash
uvicorn main:app --reload
```

### Access the Application

Once running, open your browser and navigate to:

- **Web Interface:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Case Studies:** [http://127.0.0.1:8000/casestudies](http://127.0.0.1:8000/casestudies)
- **API Docs (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📊 API Endpoints

### Text Analysis
```http
POST /analyze-text
Content-Type: application/json

{
  "text": "Article text to analyze..."
}
```

**Response:**
```json
{
  "label": "REAL" or "FAKE",
  "confidence": 95.2,
  "reason": "Analysis explanation...",
  "key_signals": ["signal1", "signal2", ...]
}
```

### Image Deepfake Detection
```http
POST /analyze-image
Content-Type: multipart/form-data

file: <image file>
```

**Response:**
```json
{
  "label": "REAL" or "FAKE",
  "confidence": 87.5,
  "reason": "Analysis explanation..."
}
```

---

## 🏗️ Project Structure

```
verifAI/
├── main.py                 # FastAPI backend server
├── requirements.txt        # Python dependencies
├── static/
│   ├── index.html         # Main web interface
│   └── casestudies.html   # Real-world incident cases
└── README.md              # This file
```

---

## 🔧 Technology Stack

- **Backend:** FastAPI, PyTorch, Transformers (Hugging Face)
- **Frontend:** HTML5, CSS3, JavaScript (vanilla)
- **Models:** Pretrained NLP and vision models from Hugging Face Model Hub
- **Charts:** Chart.js (for case studies visualization)

---

## 📚 Models Used

- **Fake News Detection:** [codegoat766/fake-news-detector](https://huggingface.co/codegoat766/fake-news-detector)
- **Deepfake Detection:** [codegoat766/deepfake-detector](https://huggingface.co/codegoat766/deepfake-detector)

---

## 📖 How It Works

1. **User Input:** Submit article text or upload an image
2. **Model Inference:** The appropriate ML model processes the input
3. **Confidence Scoring:** Returns authenticity confidence (0-100%)
4. **Signal Analysis:** Extracts key linguistic or visual features supporting the verdict
5. **Results Display:** Shows verdict, confidence, reasoning, and detected signals

---

## ⚖️ Disclaimer

VerifAI uses machine learning models that are not perfect. Results should be considered as one input among many when evaluating content authenticity. Always cross-reference with multiple sources and fact-checking organizations for critical decisions.

---

## 📄 License

This project is open source and available under the MIT License.
