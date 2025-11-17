🏥 Medical AI Chatbot
An AI-powered medical assistant built using Streamlit, Retrieval-Augmented Generation (RAG), and multiple LLM providers (Groq, OpenAI, Gemini).
Supports document ingestion, web search, voice assistant (local only), and medical guidance.

📌 Features
✅ Medical Chat Interface
Ask any medical question and get concise or detailed AI-generated answers.

📄 Document-Based RAG
Upload PDF, TXT, DOCX medical documents to build a custom knowledge base.

🌐 Web Search (Optional)
Fetch real-time medical information from trusted websites.

🤖 Multiple LLM Providers
Supports:
Groq
Google Gemini

🔊 Voice Assistant (Local only)
Cloud deployment disables TTS and STT for stability.

📁 Project Structure
Chatbot-Medics-main/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # All required Python libraries
├── medical_vectordb11.pkl  # Vector DB (auto-created after doc processing)
│
├── config/                 # API keys & configuration
├── models/                 # LLM, Embeddings, RAG components
├── utils/                  # Helper utilities (FileHandler, TextChunker, etc.)
├── uploads/                # Temporarily stores uploaded files
└── README.md               # Documentation

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/nagavibhar-web/ChatBot_NeoStats.git
cd Chatbot-Medics-main

2️⃣ Create and Activate a Virtual Environment
Windows
python -m venv venv
.\venv\Scripts\activate

Mac/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Requirements
Make sure you're inside the project folder.
pip install -r requirements.txt

4️⃣ Add API Keys (VERY IMPORTANT)
Inside config/config.py, fill in:
GROQ_API_KEY = "your-key"
GOOGLE_API_KEY = "your-key"
BRAVE_API_KEY = "your-key"   # Optional for web search

▶️ Run the Chatbot
Using VS Code Terminal
Navigate to the folder containing app.py.
streamlit run app.py
OR 
if multiple Python versions exist:
python -m streamlit run app.py

Streamlit will open on:
http://localhost:8501

📚 Knowledge Base Usage
Upload Documents
Go to the sidebar → Upload Medical Document

Supported: PDF, DOCX, TXT

Then click:
📚 Process Document


This:
Extracts text
Splits into chunks
Creates embeddings
Saves vector DB as:
medical_vectordb11.pkl

To reload later
Click:
📖 Load Existing Knowledge Base

🔧 Troubleshooting
❌ ImportError: module not found

Check folder structure; all subfolders (config, models, utils) must be present with __init__.py.

❌ API keys not working

Ensure keys are correct and active.

❌ Streamlit not opening

Try:

python -m streamlit run app.py

❌ requirements.txt fails on Python 3.14

Some packages don’t yet support Python 3.14.
Install Python 3.11 instead.

⚠️ Medical Disclaimer

This chatbot is for educational purposes only.
It is NOT a substitute for professional diagnosis or treatment.