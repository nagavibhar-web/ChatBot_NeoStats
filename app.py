import streamlit as st # type: ignore
import os
from typing import List, Dict, Any
from datetime import datetime
import logging

# Force CPU usage for Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import custom modules with error handling
try:
    from config.config import config
    from models.llm import llm_manager
    from models.embeddings import VectorDatabase, RAGRetriever, EmbeddingModel
    from utils import (
        DocumentProcessor, TextChunker, WebSearcher, VoiceAssistant,
        ResponseFormatter, SessionManager, FileHandler
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are available in your Streamlit Cloud deployment.")
    st.stop()

class MedicalChatbot:
    """Main medical chatbot class integrating all components"""
    
    def __init__(self):
        # Initialize components with error handling for Streamlit Cloud
        try:
            self.rag_retriever = RAGRetriever()
        except Exception as e:
            st.warning(f"RAG retriever initialization warning: {e}")
            self.rag_retriever = None
        
        try:
            self.web_searcher = WebSearcher()
        except Exception as e:
            st.warning(f"Web searcher initialization warning: {e}")
            self.web_searcher = None
        
        # Voice Assistant - disable for Streamlit Cloud
        try:
            self.voice_assistant = VoiceAssistant()
        except Exception as e:
            st.info("Voice features disabled for cloud deployment")
            # Create mock voice assistant for compatibility
            self.voice_assistant = self._create_mock_voice_assistant()
        
        try:
            self.document_processor = DocumentProcessor()
        except Exception as e:
            st.warning(f"Document processor initialization warning: {e}")
            self.document_processor = None
        
        try:
            self.text_chunker = TextChunker()
        except Exception as e:
            st.warning(f"Text chunker initialization warning: {e}")
            self.text_chunker = None
    
    def _create_mock_voice_assistant(self):
        """Create mock voice assistant for cloud deployment"""
        class MockVoiceAssistant:
            def is_available(self):
                return False
            def listen_once(self):
                st.warning("Voice input not available in cloud deployment")
                return None
            def speak(self, text):
                st.info(f"Would speak: {text[:100]}...")
                return
            def stop_speaking(self):
                return True
        
        return MockVoiceAssistant()
        
    def initialize_rag_system(self, file_path: str = None) -> bool:
        """Initialize RAG system from file or load existing"""
        if not self.rag_retriever:
            st.error("RAG retriever not available")
            return False
            
        try:
            # Try to load existing vector database
            if self.rag_retriever.vector_db.load_from_file():
                return True
            
            # If file provided, process it
            if file_path and os.path.exists(file_path):
                return self.process_document(file_path)
            
            return False
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")
            return False
    
    def process_document(self, file_path: str) -> bool:
        """Process document and add to RAG system"""
        if not all([self.document_processor, self.text_chunker, self.rag_retriever]):
            st.error("Required components not available for document processing")
            return False
            
        try:
            # Extract text from document
            text = self.document_processor.extract_text(file_path)
            if not text:
                st.error("No text could be extracted from the document")
                return False
            
            # Chunk the text
            chunks = self.text_chunker.chunk_text(text)
            if not chunks:
                st.error("No chunks could be created from the text")
                return False
            
            # Add to vector database
            source = os.path.basename(file_path)
            success = self.rag_retriever.add_documents_from_texts(chunks, source=source)
            
            if success:
                # Save to file
                self.rag_retriever.vector_db.save_to_file()
                st.success(f"✅ Successfully processed {len(chunks)} chunks from {source}")
                return True
            
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return False
        
        return False
    
    def generate_medical_response(self, query: str, use_web_search: bool = False, response_mode: str = "detailed") -> str:
        """Generate comprehensive medical response"""
        
        try:
            # Get RAG context
            rag_context = {"context": "No relevant context found in the knowledge base.", "sources": []}
            if self.rag_retriever:
                try:
                    rag_context = self.rag_retriever.retrieve_context(query)
                except Exception as e:
                    st.warning(f"RAG context retrieval failed: {e}")
            
            # Get web search context if enabled
            web_context = ""
            if use_web_search and self.web_searcher and self.web_searcher.is_available():
                try:
                    web_context = self.web_searcher.search_medical(query)
                except Exception as e:
                    st.warning(f"Web search failed: {e}")
                    web_context = "No recent web information found."
            
            # Prepare system prompt
            system_prompt = self._get_medical_system_prompt(response_mode)
            
            # Prepare user message with context
            user_message = self._format_user_message(query, rag_context, web_context)
            
            # Generate response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Get response parameters based on mode
            response_config = config.RESPONSE_MODES.get(response_mode, config.RESPONSE_MODES["detailed"])
            
            response = llm_manager.generate_response(
                messages,
                provider=st.session_state.get('selected_provider', 'groq'),
                **response_config
            )
            
            # Format response based on mode
            formatted_response = ResponseFormatter.format_medical_response(response, response_mode)
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logging.error(error_msg)
            return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Error: {str(e)}"
    
    def _get_medical_system_prompt(self, response_mode: str) -> str:
        """Get system prompt based on response mode"""
        base_prompt = """You are a knowledgeable medical AI assistant. Your role is to:
        1. Provide accurate medical information based on the given context
        2. Analyze symptoms and provide preliminary assessments
        3. Suggest when to seek professional medical attention
        4. Always emphasize that this is not a substitute for professional medical advice
        
        Guidelines:
        - Use the provided medical context to inform your response
        - Be compassionate and understanding
        - Provide actionable advice when appropriate
        - Always recommend consulting healthcare professionals for serious concerns
        - If you don't have enough information, clearly state this limitation"""
        
        if response_mode == "concise":
            base_prompt += "\n\nResponse Style: Provide concise, focused answers. Prioritize the most important information and keep responses brief while maintaining accuracy."
        elif response_mode == "detailed":
            base_prompt += "\n\nResponse Style: Provide comprehensive, detailed responses. Include explanations, context, and thorough guidance while maintaining clarity."
        
        return base_prompt
    
    def _format_user_message(self, query: str, rag_context: Dict, web_context: str) -> str:
        """Format user message with all available context"""
        message_parts = []
        
        # Add RAG context
        if rag_context["context"] and rag_context["context"] != "No relevant context found in the knowledge base.":
            message_parts.append(f"**Medical Knowledge Base Context:**\n{rag_context['context']}")
            if rag_context["sources"]:
                message_parts.append(f"**Sources:** {', '.join(rag_context['sources'])}")
        
        # Add web context
        if web_context and web_context != "No recent web information found.":
            message_parts.append(f"**Recent Web Information:**\n{web_context}")
        
        # Add user query
        message_parts.append(f"**Patient Query:** {query}")
        
        return "\n\n".join(message_parts)

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Medical Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_sidebar() -> Dict[str, Any]:
    """Setup sidebar controls and return configuration"""
    with st.sidebar:
        st.title("🏥 Medical AI Configuration")
        
        # LLM Provider Selection
        st.subheader("🤖 AI Model")
        try:
            available_providers = llm_manager.get_available_providers()
            if available_providers:
                selected_provider = st.selectbox(
                    "Select AI Provider",
                    available_providers,
                    index=0 if st.session_state.selected_provider not in available_providers else available_providers.index(st.session_state.selected_provider)
                )
                st.session_state.selected_provider = selected_provider
            else:
                st.error("❌ No AI providers available. Check your API keys.")
                st.session_state.selected_provider = None
        except Exception as e:
            st.error(f"Error loading AI providers: {e}")
            st.session_state.selected_provider = None
        
        # Response Mode
        st.subheader("📝 Response Mode")
        try:
            response_mode = st.radio(
                "Choose response style:",
                options=list(config.RESPONSE_MODES.keys()),
                index=0 if st.session_state.response_mode == "concise" else 1,
                format_func=lambda x: f"{x.title()} - {config.RESPONSE_MODES[x]['description']}"
            )
            st.session_state.response_mode = response_mode
        except Exception as e:
            st.warning(f"Response mode configuration error: {e}")
            st.session_state.response_mode = "detailed"
            response_mode = "detailed"
        
        # Voice Settings - Modified for cloud deployment
        st.subheader("🎤 Voice Assistant")
        voice_available = (st.session_state.get('chatbot', None) and 
                          st.session_state.chatbot.voice_assistant.is_available())
        
        if not voice_available:
            st.info("Voice features are disabled in cloud deployment")
            st.session_state.voice_enabled = False
            voice_enabled = False
        else:
            voice_enabled = st.checkbox(
                "Enable Voice", 
                value=st.session_state.voice_enabled,
                disabled=True  # Always disabled for cloud
            )
            st.session_state.voice_enabled = False  # Force disabled
        
        # Web Search
        st.subheader("🌐 Web Search")
        web_search_available = (st.session_state.get('chatbot', None) and 
                               st.session_state.chatbot.web_searcher and 
                               st.session_state.chatbot.web_searcher.is_available())
        web_search_enabled = st.checkbox(
            "Enable Web Search", 
            value=st.session_state.web_search_enabled,
            disabled=not web_search_available,
            help="Search the web for recent medical information"
        )
        st.session_state.web_search_enabled = web_search_enabled and web_search_available
        
        # Document Upload
        st.subheader("📄 Knowledge Base")
        uploaded_file = st.file_uploader(
            "Upload Medical Document",
            type=['pdf', 'txt', 'docx'],
            help="Upload medical documents to enhance the AI's knowledge"
        )
        
        if uploaded_file and st.session_state.get('chatbot'):
            try:
                is_valid, message = FileHandler.validate_file(uploaded_file)
                if is_valid:
                    if st.button("📚 Process Document"):
                        with st.spinner("Processing document..."):
                            file_path = FileHandler.save_uploaded_file(uploaded_file)
                            if file_path:
                                success = st.session_state.chatbot.process_document(file_path)
                                if success:
                                    st.session_state.vector_db_initialized = True
                                # Clean up temporary file
                                try:
                                    os.remove(file_path)
                                except:
                                    pass
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"File handling error: {e}")
        
        # Load existing database
        if st.button("📖 Load Existing Knowledge Base"):
            if st.session_state.get('chatbot') and st.session_state.chatbot.initialize_rag_system():
                st.session_state.vector_db_initialized = True
                st.success("✅ Knowledge base loaded!")
            else:
                st.error("❌ No existing knowledge base found")
        
        st.divider()
        
        # System Status
        st.subheader("📊 System Status")
        
        # RAG Status
        if st.session_state.vector_db_initialized and st.session_state.get('chatbot'):
            try:
                if st.session_state.chatbot.rag_retriever:
                    stats = st.session_state.chatbot.rag_retriever.get_database_stats()
                    st.success(f"📚 Knowledge Base: {stats['total_chunks']} chunks")
                    if stats['sources']:
                        st.info(f"📁 Sources: {', '.join(stats['sources'][:3])}{'...' if len(stats['sources']) > 3 else ''}")
                else:
                    st.warning("📚 Knowledge Base: Not available")
            except Exception as e:
                st.warning(f"📚 Knowledge Base: Error reading stats - {e}")
        else:
            st.warning("📚 Knowledge Base: Not loaded")
        
        # AI Provider Status
        if st.session_state.selected_provider:
            st.success(f"🤖 AI: {st.session_state.selected_provider.title()}")
        else:
            st.error("🤖 AI: Not available")
        
        # Voice Status - Always disabled for cloud
        st.info("🎤 Voice: Disabled (Cloud deployment)")
        
        # Web Search Status
        search_status = "✅ Enabled" if st.session_state.web_search_enabled else "❌ Disabled"
        st.info(f"🌐 Web Search: {search_status}")
        
        st.divider()
        
        # Conversation Management
        st.subheader("💬 Conversation")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                SessionManager.clear_messages()
                st.rerun()
        
        with col2:
            if st.button("💾 Export", use_container_width=True):
                try:
                    export_text = SessionManager.export_conversation()
                    st.download_button(
                        "📥 Download",
                        export_text,
                        file_name=f"medical_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        return {
            'provider': st.session_state.selected_provider,
            'response_mode': response_mode,
            'voice_enabled': False,  # Always False for cloud
            'web_search_enabled': st.session_state.web_search_enabled
        }

def instructions_page():
    """Instructions and setup page"""
    st.title("🏥 Medical Chatbot - Setup Guide")
    st.markdown("### AI-Powered Medical Consultation with Voice, RAG, and Web Search")
    
    # Cloud deployment notice
    st.info("📢 **Cloud Deployment Notice**: Voice features are disabled in this cloud deployment for compatibility. Text-based interaction is fully available.")
    
    # Quick Start
    st.markdown("## 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1. Setup APIs** 🔑
        - Add API keys to config
        - Groq/OpenAI/Gemini supported
        """)
    
    with col2:
        st.markdown("""
        **2. Upload Documents** 📄
        - PDF, TXT, DOCX supported
        - Creates knowledge base
        """)
    
    with col3:
        st.markdown("""
        **3. Start Chatting** 💬
        - Text input available
        - AI-powered responses
        """)
    
    # Features
    st.markdown("## ✨ Features")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 RAG System", "🌐 Web Search", "🎤 Voice AI", "📊 Response Modes"])
    
    with tab1:
        st.markdown("""
        ### Retrieval-Augmented Generation (RAG)
        
        **What it does:**
        - Processes your medical documents
        - Creates searchable knowledge base
        - Provides context-aware responses
        
        **Supported formats:**
        - PDF (medical journals, guidelines)
        - TXT (clinical notes, research)
        - DOCX (reports, documentation)
        
        **How it works:**
        1. Documents are chunked intelligently
        2. Text converted to vector embeddings
        3. Relevant chunks retrieved for each query
        4. AI generates responses using your data
        """)
    
    with tab2:
        st.markdown("""
        ### Live Web Search Integration
        
        **Capabilities:**
        - Real-time medical information
        - Latest research and guidelines
        - Trusted medical sources priority
        
        **Sources include:**
        - Mayo Clinic
        - WebMD
        - Healthline
        - Medical journals
        
        **When it helps:**
        - Recent medical developments
        - Current treatment guidelines
        - Latest research findings
        - Drug information updates
        """)
    
    with tab3:
        st.markdown("""
        ### Voice Assistant (Local Deployment Only)
        
        **Note:** Voice features are disabled in cloud deployments for compatibility.
        
        **Input Features:**
        - Speech-to-text conversion
        - Natural language processing
        - Timeout and error handling
        
        **Output Features:**
        - Text-to-speech responses
        - Adjustable speech rate
        - Stop/pause controls
        
        **Voice Commands:**
        - 🤒 "Ask about symptoms"
        - 💊 "Ask about medication" 
        - 🩺 "General health question"
        """)
    
    with tab4:
        st.markdown("""
        ### Response Modes
        
        **Concise Mode:**
        - Short, focused answers
        - Key points only
        - Quick consultations
        - Mobile-friendly
        
        **Detailed Mode:**
        - Comprehensive responses
        - Full explanations
        - Context and background
        - In-depth analysis
        
        **Automatic Formatting:**
        - Medical terminology explained
        - Structured information
        - Clear recommendations
        - Professional disclaimers
        """)
    
    # API Setup
    st.markdown("## 🔑 API Keys Setup")
    
    with st.expander("📋 Required API Keys", expanded=True):
        st.markdown("""
        ### LLM Providers (Choose one or more)
        
        **Groq (Recommended)**
        ```python
        GROQ_API_KEY = "your-groq-api-key"
        ```
        - Get key: [console.groq.com](https://console.groq.com/keys)
        - Fast inference
        - Cost-effective
        
        **OpenAI**
        ```python
        OPENAI_API_KEY = "your-openai-key"
        ```
        - Get key: [platform.openai.com](https://platform.openai.com/api-keys)
        - High quality responses
        
        **Google Gemini**
        ```python
        GOOGLE_API_KEY = "your-google-key"  
        ```
        - Get key: [aistudio.google.com](https://aistudio.google.com/app/apikey)
        
        ### Web Search (Optional)
        **Brave Search**
        ```python
        BRAVE_API_KEY = "your-brave-key"
        ```
        - Get key: [brave.com/search/api](https://brave.com/search/api/)
        - Real-time web search
        """)
    
    # Technical Details
    st.markdown("## ⚙️ Technical Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Components:**
        - 🧠 Vector Database (Sentence Transformers)
        - 🔍 Similarity Search (Cosine)
        - 🤖 Multiple LLM Support
        - 🎤 Speech Recognition (Local only)
        - 🔊 Text-to-Speech (Local only)
        - 🌐 Web Search (Brave API)
        """)
    
    with col2:
        st.markdown("""
        **Processing Pipeline:**
        1. Document ingestion & chunking
        2. Embedding generation
        3. Vector storage & indexing
        4. Query processing
        5. Context retrieval
        6. Response generation
        7. Voice synthesis (local only)
        """)
    
    # Troubleshooting
    st.markdown("## 🔧 Troubleshooting")
    
    with st.expander("Common Issues & Solutions"):
        st.markdown("""
        **API Key Errors:**
        - Verify keys are correct and active
        - Check API quotas and billing
        - Ensure proper environment setup
        
        **Document Processing Issues:**
        - Check file format (PDF/TXT/DOCX)
        - Ensure files contain readable text
        - Verify file size < 10MB
        
        **Cloud Deployment Issues:**
        - Voice features are disabled by design
        - Ensure all dependencies are in requirements.txt
        - Check Streamlit Cloud logs for errors
        
        **Performance Issues:**
        - Large documents take time to process
        - Reduce chunk size for faster processing
        - Clear browser cache if needed
        """)
    
    # Best Practices
    st.markdown("## 💡 Best Practices")
    
    st.markdown("""
    ### For Medical Professionals
    - Upload current clinical guidelines and protocols
    - Include drug interaction databases
    - Add institution-specific procedures
    - Regular knowledge base updates
    
    ### For Patients
    - Always verify AI advice with healthcare providers
    - Use for educational purposes only
    - Provide clear, specific symptoms descriptions
    - Include relevant medical history context
    
    ### For Optimal Performance
    - Use high-quality, text-based documents
    - Organize documents by medical specialty
    - Keep documents updated and relevant
    - Monitor cloud deployment resource usage
    """)
    
    st.markdown("---")
    st.markdown("**Ready to start?** Navigate to the **Medical Chat** page! 🚀")

def chat_page():
    """Main medical chat interface"""
    st.title("🏥 Medical Chatbot")
    
    # Setup sidebar and get configuration
    config_dict = setup_sidebar()
    
    # Check system readiness
    if not st.session_state.selected_provider:
        st.error("❌ No AI provider available. Please check your API keys in the config.")
        return
    
    if not st.session_state.vector_db_initialized:
        st.warning("⚠️ Knowledge base not loaded. Upload documents or load existing database for enhanced responses.")
        st.info("💡 The AI can still answer general medical questions using its training data.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Text input only for cloud deployment
    prompt = st.chat_input("Type your medical question here...")
    
    # Process text input
    if prompt:
        process_user_input(prompt, config_dict)
        st.rerun()
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🤒 Symptom Analysis", use_container_width=True):
            quick_question = "I want to discuss my symptoms and get a preliminary analysis."
            process_user_input(quick_question, config_dict)
            st.rerun()
    
    with col2:
        if st.button("💊 Medication Info", use_container_width=True):
            quick_question = "I need information about medications, dosages, or drug interactions."
            process_user_input(quick_question, config_dict)
            st.rerun()
    
    with col3:
        if st.button("🩺 Health Guidance", use_container_width=True):
            quick_question = "I have a general health question and need medical guidance."
            process_user_input(quick_question, config_dict)
            st.rerun()
    
    with col4:
        if st.button("🚨 Emergency Info", use_container_width=True):
            quick_question = "When should I seek immediate medical attention for my condition?"
            process_user_input(quick_question, config_dict)
            st.rerun()
    
    # Medical Disclaimer
    st.markdown("---")
    st.markdown("""
<div style="background-color: #fff8e1; color: #333333; border: 1px solid #ffcc80; border-radius: 5px; padding: 10px; margin: 10px 0;">
<strong>⚠️ Medical Disclaimer:</strong><br>
This AI assistant provides general medical information for educational purposes only. 
It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay seeking it because of something you have read here.
</div>
""", unsafe_allow_html=True)

def process_user_input(user_input: str, config_dict: Dict[str, Any]):
    """Process user input and generate response"""
    if not user_input.strip():
        return
    
    # Add user message
    SessionManager.add_message("user", user_input)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your question..."):
            try:
                response = st.session_state.chatbot.generate_medical_response(
                    user_input,
                    use_web_search=config_dict['web_search_enabled'],
                    response_mode=config_dict['response_mode']
                )
                
                st.markdown(response)
                
                # Add response to session
                SessionManager.add_message("assistant", response)
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                SessionManager.add_message("assistant", error_msg)

def main():
    """Main application function"""
    # Setup page
    setup_page_config()
    
    # Initialize session
    SessionManager.initialize_session()
    
    # Initialize chatbot with error handling
    if 'chatbot' not in st.session_state:
        with st.spinner("🤖 Initializing Medical AI Assistant..."):
            try:
                st.session_state.chatbot = MedicalChatbot()
                
                # Try to load existing knowledge base
                if st.session_state.chatbot.initialize_rag_system():
                    st.session_state.vector_db_initialized = True
                    
            except Exception as e:
                st.error(f"Failed to initialize Medical Chatbot: {e}")
                st.error("Please check your configuration and try refreshing the page.")
                st.stop()
    
    # Navigation
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Select Page:",
            ["🏥 Medical Chat", "📖 Instructions"],
            index=0,
            key="navigation"
        )
    
    # Route to appropriate page
    if "Instructions" in page:
        instructions_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()



