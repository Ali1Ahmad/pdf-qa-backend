from flask import Flask, request, jsonify  # Flask for API creation
from flask_cors import CORS  # Allows cross-origin access (e.g., frontend to backend)
from werkzeug.utils import secure_filename  # Prevents file path exploits for uploads
import os
from dotenv import load_dotenv  # Loads variables from .env
import requests  # For calling Groq API
import pdfplumber  # Extracts text from PDF files
from chromadb import Client  # ChromaDB client for vector search
from chromadb.config import Settings  # ChromaDB configuration
from datetime import datetime  # Used for timestamps
import re  # Regex for text cleaning

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for frontend running at localhost
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"], 
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configurations for file uploads
app.config['UPLOAD_FOLDER'] = 'app/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('db', exist_ok=True)

# Load API key from environment
API_KEY = os.getenv('GROQ_API_KEY')
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize ChromaDB client using cosine similarity
chroma_client = Client(Settings(
    persist_directory="chroma_db",
    anonymized_telemetry=False
))

# Create or load ChromaDB collection
# Used to store and query PDF chunks
def init_chroma():
    global collection
    try:
        collection = chroma_client.get_or_create_collection(
            name="pdf-data",
            metadata={"hnsw:space": "cosine"}  # use cosine similarity
        )
        print("\u2705 ChromaDB Collection Ready")
    except Exception as error:
        print("\u274C ChromaDB Connection Failed:", error)
        exit(1)

init_chroma()

# Check if the file extension is allowed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Split long PDF text into overlapping chunks for vector storage

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        if start > 0:
            start = start - overlap  # create overlap with previous chunk
        if end >= text_length:
            chunks.append(text[start:])  # last chunk
            break
        last_space = text.rfind(' ', start, end)  # avoid splitting words
        if last_space != -1:
            end = last_space
        chunks.append(text[start:end])
        start = end
    return chunks

# PDF Upload Endpoint
# Accepts PDF file, extracts text, chunks it, and stores it in ChromaDB
@app.route('/upload', methods=['POST'])
def upload_file():
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid or no file selected'}), 400

        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        text = ""
        metadata = {
            "filename": filename,
            "page_count": 0,
            "upload_time": datetime.now().isoformat()
        }

        # Extract text from PDF using pdfplumber
        with pdfplumber.open(filepath) as pdf:
            metadata["page_count"] = len(pdf.pages)
            pages_text = [f"[Page {i+1}] {page.extract_text() or ''}" for i, page in enumerate(pdf.pages)]
            text = "\n\n".join(pages_text)

        if not text.strip():
            return jsonify({'error': 'No text could be extracted from the PDF'}), 400

        # Split PDF text into overlapping chunks
        chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)
        chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

        # Delete old version of this file if it exists
        try:
            collection.delete()
        except:
            pass

        # Store chunks in ChromaDB
        collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=[{**metadata, "chunk_index": i} for i in range(len(chunks))]
        )

        return jsonify({
            'message': 'PDF processed successfully',
            'metadata': metadata,
            'text_length': len(text),
            'chunks_count': len(chunks)
        })
    except Exception as error:
        return jsonify({'error': 'Failed to process PDF'}), 500
    finally:
        # Clean up saved file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

# Query Endpoint
# Accepts a question and returns an answer using relevant PDF content
@app.route('/query', methods=['POST'])
def query_pdf():
    try:
        data = request.get_json()
        if not data or not data.get('question'):
            return jsonify({'error': 'No question provided'}), 400

        question = data['question']
        print(f"\nReceived question: {question}")

        # Search for top 4 most similar chunks using cosine similarity
        results = collection.query(
            query_texts=[question],
            n_results=4,
            include=['metadatas', 'distances', 'documents']
        )

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        if not documents:
            return jsonify({'answer': 'No relevant information found', 'context': None})

        # FIX 1: Use a more lenient similarity threshold (was 1.2 before)
        SIMILARITY_THRESHOLD = 0.95
        if distances[0] > SIMILARITY_THRESHOLD:
            return jsonify({
                'answer': 'No sufficiently relevant information found',
                'metadata': metadatas[0],
                'relevance_score': 1 - distances[0]
            })

        # FIX 2: Combine top 3 matching chunks for broader context
        top_chunks = documents[:3]
        combined_text = "\n\n---\n\n".join(top_chunks)
        metadata = metadatas[0]
        relevance_score = 1 - distances[0]

        # FIX 3: Relax the prompt to let the model try its best
        system_prompt = f"""Answer the question using ONLY the following content.\n
Content from PDF '{metadata.get('filename', 'unknown')}' (uploaded {metadata.get('upload_time', 'unknown')}):\n
{combined_text}\n\nTry your best to answer based on the above content. If the answer truly cannot be found, say so."""

        # Call Groq API to generate the answer
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},  # To Do loged data input last 5 items
                    {"role": "user", "content": question}
                ],
                "temperature": 0.1  # low temp = more focused answer
            },
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        response.raise_for_status()
        response_data = response.json()

        # Return LLM answer with metadata
        return jsonify({
            'answer': response_data['choices'][0]['message']['content'],
            'metadata': metadata,
            'relevance_score': relevance_score
        })

    except Exception as error:
        return jsonify({'error': f'Failed to query PDF: {str(error)}'}), 500

# Run the Flask app locally
if __name__ == '__main__':
    app.run(port=5050, debug=True)