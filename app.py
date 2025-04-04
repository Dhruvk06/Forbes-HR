import os
import requests
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import google.generativeai as genai
from flask_cors import CORS
# Load environment variables
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip('/')
PSFT_USER = os.getenv("PSFT_USER")
PSFT_PASS = os.getenv("PSFT_PASS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp-01-21')

app = Flask(__name__)
CORS(app)

def get_employee_data(token):
    url = f"{API_BASE_URL}/{token}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    try:
        response = requests.get(url, auth=HTTPBasicAuth(PSFT_USER, PSFT_PASS), headers=headers, timeout=5)
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
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    token = request.form.get('token')
    question = request.form.get('question')
    
    if not token or not question:
        return jsonify({'response': 'Missing token or question'}), 400
    
    data = get_employee_data(token)
    if isinstance(data, str):
        return jsonify({'response': data})
    
    try:
        prompt = f"Based on this employee data:\n{data}\n\nQuestion: {question}\nAnswer:"
        response = model.generate_content(prompt)
        return jsonify({'response': response.text.strip()})
    except Exception as e:
        return jsonify({'response': f"AI Error: {e}"}), 500

# Route to serve the chat widget JavaScript file
@app.route('/chat-widget.js')
def serve_widget():
    return send_file("static/chat-widget.js", mimetype="application/javascript")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
