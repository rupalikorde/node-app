import os
import tempfile
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import Client
import uuid
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize Gemini
gemini_api_key = os.getenv('GEMINI_API_KEY')
if gemini_api_key and gemini_api_key != 'your_gemini_api_key_here':
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    print("✅ Gemini AI configured")
else:
    model = None
    print("⚠️ Gemini API key not configured - AI features will be limited")

# Initialize LangSmith (optional)
langsmith_client = None
try:
    if os.getenv('LANGCHAIN_API_KEY'):
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_PROJECT'] = 'RAG-App'
        langsmith_client = Client()
        print("✅ LangSmith monitoring enabled")
    else:
        print("⚠️ LangSmith API key not found - monitoring disabled")
except Exception as e:
    print(f"⚠️ LangSmith initialization failed: {e} - monitoring disabled")
    langsmith_client = None

# Initialize embeddings
embeddings = None
if gemini_api_key and gemini_api_key != 'your_gemini_api_key_here':
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        print("✅ Gemini Embeddings configured")
    except Exception as e:
        print(f"⚠️ Gemini Embeddings failed: {e}")
        print("💡 Falling back to simple text chunking without embeddings")
        embeddings = None
else:
    print("⚠️ Gemini API key not configured - using simple text search fallback")

# Global vector store
vector_store = None
documents = []

def safe_langsmith_log(name, run_type, inputs, outputs, project_name="RAG-App"):
    """Safely log to LangSmith if available"""
    try:
        if langsmith_client:
            langsmith_client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                outputs=outputs,
                project_name=project_name
            )
            return True
    except Exception as e:
        print(f"⚠️ LangSmith logging failed: {e}")
    return False

class RAGSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_document(self, file_path, file_type):
        """Process uploaded document and create chunks"""
        try:
            print(f"📄 Processing {file_type} file: {file_path}")
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_type == 'txt':
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} document(s)")
            chunks = self.text_splitter.split_documents(documents)
            print(f"✅ Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            return []
    
    def create_vector_store(self, chunks):
        """Create FAISS vector store from document chunks"""
        try:
            if not embeddings:
                print("⚠️ Embeddings not available - storing chunks for simple text search")
                print("   Vector search disabled, using keyword matching instead")
                # Return chunks directly for simple text search
                return chunks
            if chunks:
                print(f"✅ Creating vector store with {len(chunks)} chunks...")
                vector_store = FAISS.from_documents(chunks, embeddings)
                print("✅ Vector store created successfully!")
                return vector_store
            else:
                print("⚠️ No chunks provided for vector store creation")
            return None
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            print("💡 Falling back to simple text search...")
            # Fallback to simple text chunks
            return chunks if chunks else None
    
    def retrieve_context(self, query, vector_store, k=3):
        """Retrieve relevant context from vector store or simple text search"""
        try:
            if not vector_store:
                return ""
            
            # Check if vector_store is actually a FAISS vector store or just chunks
            if hasattr(vector_store, 'similarity_search'):
                # Use vector similarity search
                docs = vector_store.similarity_search(query, k=k)
                context = "\n\n".join([doc.page_content for doc in docs])
                return context
            elif isinstance(vector_store, list):
                # Fallback: simple keyword search in chunks
                print(f"🔍 Using simple text search for query: {query}")
                query_words = query.lower().split()
                scored_chunks = []
                
                for chunk in vector_store:
                    content = chunk.page_content.lower()
                    # Simple scoring based on keyword matches
                    score = sum(word in content for word in query_words)
                    if score > 0:
                        scored_chunks.append((score, chunk.page_content))
                
                # Sort by score and take top k
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                top_chunks = [chunk[1] for chunk in scored_chunks[:k]]
                context = "\n\n".join(top_chunks)
                return context
            
            return ""
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
    
    def generate_response(self, query, context):
        """Generate response using Gemini with context"""
        try:
            if not model:
                return "⚠️ Gemini AI is not configured. Please add your GEMINI_API_KEY to the .env file to enable AI responses."
            
            prompt = f"""
            Context: {context}
            
            Question: {query}
            
            Please answer the question based on the provided context. If the context doesn't contain relevant information, please say so.
            """
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response at this time. Please check your API configuration."

# Initialize RAG system
rag_system = RAGSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store, documents
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        allowed_extensions = {'txt', 'pdf'}
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        if file_extension not in allowed_extensions:
            return jsonify({'error': 'Only TXT and PDF files are supported'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
            file.save(temp_file.name)
            
            # Process document
            chunks = rag_system.process_document(temp_file.name, file_extension)
            
            if chunks:
                # Create vector store
                new_vector_store = rag_system.create_vector_store(chunks)
                
                if new_vector_store:
                    # Only update global state if vector store creation succeeded
                    vector_store = new_vector_store
                    documents = chunks
                    
                    # Log to LangSmith (optional)
                    safe_langsmith_log(
                        name="document_upload",
                        run_type="tool",
                        inputs={"filename": filename, "chunks_count": len(chunks)},
                        outputs={"status": "success"}
                    )
                    
                    os.unlink(temp_file.name)  # Clean up temp file
                    return jsonify({
                        'message': f'Document processed successfully! Created {len(chunks)} chunks.',
                        'chunks_count': len(chunks)
                    })
                else:
                    os.unlink(temp_file.name)
                    return jsonify({'error': 'Failed to create vector store. Please check your Gemini API key configuration.'}), 500
            else:
                os.unlink(temp_file.name)
                return jsonify({'error': 'Failed to process document'}), 500
                
    except Exception as e:
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global vector_store
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if not vector_store:
            return jsonify({'error': 'Please upload a document first'}), 400
        
        # Start LangSmith run
        run_id = str(uuid.uuid4())
        
        # Retrieve context
        context = rag_system.retrieve_context(query, vector_store)
        
        # Generate response
        response = rag_system.generate_response(query, context)
        
        # Log to LangSmith (optional)
        safe_langsmith_log(
            name="rag_query",
            run_type="chain",
            inputs={"query": query, "context": context[:500] + "..." if len(context) > 500 else context},
            outputs={"response": response}
        )
        
        return jsonify({
            'response': response,
            'context_used': bool(context)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

@app.route('/status')
def status():
    return jsonify({
        'vector_store_ready': vector_store is not None,
        'document_chunks': len(documents) if documents else 0,
        'gemini_configured': model is not None,
        'embeddings_configured': embeddings is not None,
        'langsmith_configured': langsmith_client is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
