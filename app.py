import os
import requests
import fitz
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import google.generativeai as genai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import bleach
import pickle
import re

nltk.download('punkt', quiet=True)
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip('/')
PSFT_USER = os.getenv("PSFT_USER")
PSFT_PASS = os.getenv("PSFT_PASS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_PATH = "HR POLICY MANUAL - India.pdf"
FAISS_INDEX_PATH = "pdf_index.faiss"
CHUNKS_METADATA_PATH = "pdf_chunks.pkl"

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp-01-21')

app = Flask(__name__)
CORS(app)

# Initialize SentenceTransformer only when needed
embedder = None

def init_embedder():
    """Initialize SentenceTransformer lazily."""
    global embedder
    if embedder is None:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder

def format_policy_response(text):
    """Format text as a plain bulleted list, removing all markdown and page references."""
    # Remove page references (e.g., "Page 78:")
    text = re.sub(r'Page \d+:\s*', '', text)
    # Remove all markdown (bold **text**, italic *text*, standalone *)
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    text = re.sub(r'^\s*\*+\s*', '', text, flags=re.MULTILINE)  # Remove leading asterisks
    text = re.sub(r'\s*\*+\s*', ' ', text)  # Remove inline asterisks
    # Normalize whitespace
    text = ' '.join(text.split())
    # Split into sentences, ignoring empty ones
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    # Format as bulleted list with plain '-'
    if sentences:
        return '\n'.join(f"- {sentence}{'' if sentence.endswith('.') else '.'}" for sentence in sentences)
    return "- No details available."

def vectorize_pdf():
    """Vectorize the HR Policy Manual PDF and save to FAISS index."""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")
    
    doc = fitz.open(PDF_PATH)
    chunks, metadata = [], []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        if text:
            sentences = sent_tokenize(text)
            for j, sent in enumerate(sentences):
                if len(sent) > 20:  # Filter short sentences
                    chunks.append(sent)
                    metadata.append({"page": i + 1, "sentence": j})
    doc.close()
    
    # Initialize embedder if not already done
    embedder = init_embedder()
    
    # Encode chunks into vectors
    vectors = embedder.encode(chunks)
    faiss.normalize_L2(vectors)
    
    # Create and save FAISS index
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    
    # Save chunks and metadata
    with open(CHUNKS_METADATA_PATH, 'wb') as f:
        pickle.dump((chunks, metadata), f)
    
    return faiss_index, chunks, metadata

# Load or create FAISS index only if files don't exist
faiss_index, chunks, metadata = None, None, None
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_METADATA_PATH) and os.path.exists(PDF_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_METADATA_PATH, 'rb') as f:
        chunks, metadata = pickle.load(f)
else:
    faiss_index, chunks, metadata = vectorize_pdf()

def search_pdf(question):
    """Search the PDF FAISS index for relevant content."""
    # Initialize embedder if not already done
    embedder = init_embedder()
    
    query = embedder.encode([question])
    faiss.normalize_L2(query)
    D, I = faiss_index.search(query, 5)  # Top 5 results
    results = []
    for i, idx in enumerate(I[0]):
        if D[0][i] > 0.2:  # Relevance threshold
            results.append(chunks[idx])  # Return sentence without page number
    return '\n'.join(results) or "No relevant PDF data found."

def get_employee_data(token):
    """Fetch employee data from PeopleSoft API."""
    url = f"{API_BASE_URL}/{bleach.clean(token.strip())}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(PSFT_USER, PSFT_PASS),
            headers=headers,
            timeout=5,
            verify=False
        )
        if response.status_code == 401:
            return "Authentication Failed"
        elif response.status_code == 403:
            return "Access Denied"
        elif response.status_code == 404:
            return "Invalid Token"
        elif response.status_code >= 500:
            return "Server Error"
        
        data = response.json()
        return data.get('FM_EMPL_INFO_RESP', {})
    except requests.exceptions.RequestException as err:
        return f"API Error: {err}"

@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle PeopleSoft API queries."""
    token = request.form.get('token')
    question = request.form.get('question')
    
    if not token or not question:
        return jsonify({'response': 'Missing token or question'}), 400
    
    data = get_employee_data(token)
    if isinstance(data, str):
        return jsonify({'response': data})
    
    try:
        prompt = f"""
        You are an HR assistant answering questions based on employee data from PeopleSoft.
        Employee Data: {data}
        Question: {question}
        Provide a concise and accurate answer based only on the provided data. If the answer is not available, say "Information not available."
        """
        response = model.generate_content(prompt)
        return jsonify({'response': response.text.strip()})
    except Exception as e:
        return jsonify({'response': f"AI Error: {e}"}), 500

@app.route('/hr_policy_query', methods=['POST'])
def hr_policy_query():
    """Handle HR Policy Manual queries using RAG."""
    question = request.form.get('question')
    
    if not question:
        return jsonify({'response': 'Missing question'}), 400
    
    pdf_info = search_pdf(question)
    formatted_pdf_info = format_policy_response(pdf_info)
    
    try:
        prompt = f"""
        You are an HR assistant answering questions based on the Forbes Marshall HR Policy Manual.
        Question: {question}
        Relevant Policy Information: {formatted_pdf_info}
        Provide a concise and accurate answer in a plain text bulleted list format using '-' as the bullet character. Do not use any markdown formatting (e.g., no asterisks *, bold **, or italics). Do not include page references. If the answer is not available, return a single bullet saying "Information not available in the policy manual."
        """
        response = model.generate_content(prompt)
        formatted_response = format_policy_response(response.text.strip())
        return jsonify({'response': formatted_response})
    except Exception as e:
        return jsonify({'response': f"AI Error: {e}"}), 500

@app.route('/chat-widget.js')
def serve_widget():
    """Serve the chat widget JavaScript file."""
    return send_file("static/chat-widget.js", mimetype="application/javascript")

if __name__ == '__main__':
    app.run(debug=True, port=5000)