# VerifAI

VerifAI is a web application that provides AI-powered detection for:
1. **Fake News** (Text Classification)
2. **Deepfakes** (Image Classification)

This project runs entirely on a **FastAPI** backend and serves a lightweight HTML frontend. It downloads and uses pre-trained PyTorch models hosted on Hugging Face.

## Requirements

- Python 3.9+
- `pip` (Python package manager)

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/codegoat766/verifAI.git
   cd verifAI
   ```

2. (Optional but recommended) Create and activate a virtual environment:
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

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the FastAPI server, run the following command in your terminal from the project root:

```bash
uvicorn main:app --reload
```

*Note: The first time you run this, it may take a few moments as the models (`fake-news-detector` and `deepfake-detector`) will be downloaded from Hugging Face.*

## Accessing the App

Once the server has started, you can access the application in your browser:

- **Web Frontend:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
