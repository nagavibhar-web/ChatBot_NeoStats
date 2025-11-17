import os
import re
import tempfile
import PyPDF2 # type: ignore
import docx # type: ignore
import requests # type: ignore
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import streamlit as st # type: ignore
import speech_recognition as sr # type: ignore
import pyttsx3 # type: ignore
import threading
import time
from bs4 import BeautifulSoup # type: ignore
from urllib.parse import quote_plus, urljoin
from config.config import config

class DocumentProcessor:
    """Handles document processing for different file formats"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        st.warning(f"Error reading page {page_num + 1}: {e}")
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
        return text
    
    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(txt_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                st.error(f"Error reading TXT file: {str(e)}")
                return ""
        except Exception as e:
            st.error(f"Error reading TXT file: {str(e)}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return cls.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return cls.extract_text_from_txt(file_path)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return ""

class TextChunker:
    """Handles text chunking for RAG"""
    
    @staticmethod
    def chunk_text(text: str, 
                   chunk_size: int = None, 
                   overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        if not text or len(text) < chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end chunk at sentence boundary
            if end < len(text):
                # Look for sentence ending
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_boundary = max(last_period, last_newline)
                
                if last_boundary > chunk_size * 0.7:  # If boundary is not too early
                    chunk = chunk[:last_boundary + 1]
                    end = start + last_boundary + 1
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    @staticmethod
    def chunk_by_paragraphs(text: str, max_chunk_size: int = None) -> List[str]:
        """Chunk text by paragraphs, combining small ones"""
        max_chunk_size = max_chunk_size or config.CHUNK_SIZE
        
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

class WebSearcher:
    """Handles web search functionality using free DuckDuckGo search"""
    
    def __init__(self, api_key: str = None):
        # Keep the api_key parameter for compatibility but don't use it
        self.api_key = api_key or config.BRAVE_API_KEY
        self.search_enabled = config.SEARCH_ENABLED  # Don't depend on API key
        self.session = requests.Session()
        # Set user agent to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def is_available(self) -> bool:
        """Check if web search is available"""
        return self.search_enabled
    
    def search(self, query: str, num_results: int = None) -> List[Dict[str, Any]]:
        """Perform web search using DuckDuckGo"""
        if not self.is_available():
            return []
        
        num_results = num_results or config.MAX_SEARCH_RESULTS
        
        try:
            # Use DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = self.session.get(
                search_url,
                timeout=config.SEARCH_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse DuckDuckGo results
            for result_div in soup.find_all('div', class_='result')[:num_results]:
                try:
                    title_element = result_div.find('a', class_='result__a')
                    snippet_element = result_div.find('a', class_='result__snippet')
                    
                    if title_element and snippet_element:
                        title = title_element.get_text(strip=True)
                        url = title_element.get('href', '')
                        description = snippet_element.get_text(strip=True)
                        
                        # Clean up URL if it's a DuckDuckGo redirect
                        if url.startswith('/l/?uddg='):
                            try:
                                import urllib.parse
                                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                                if 'uddg' in parsed:
                                    url = urllib.parse.unquote(parsed['uddg'][0])
                            except:
                                pass
                        
                        results.append({
                            "title": title,
                            "url": url,
                            "description": description,
                            "content": description,  # Using description as content
                            "published": "",  # DuckDuckGo doesn't provide publish date easily
                            "source": "web_search"
                        })
                except Exception as e:
                    continue  # Skip malformed results
            
            # Fallback: Try alternative DuckDuckGo API if HTML parsing fails
            if not results:
                results = self._try_duckduckgo_api(query, num_results)
            
            return results
            
        except Exception as e:
            st.error(f"Web search error: {e}")
            return []
    
    def _try_duckduckgo_api(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Fallback method using DuckDuckGo instant answer API"""
        try:
            # Use DuckDuckGo instant answer API (limited but free)
            api_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
            
            response = self.session.get(api_url, timeout=config.SEARCH_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Get abstract if available
            if data.get('Abstract'):
                results.append({
                    "title": data.get('Heading', 'DuckDuckGo Result'),
                    "url": data.get('AbstractURL', ''),
                    "description": data.get('Abstract', ''),
                    "content": data.get('Abstract', ''),
                    "published": "",
                    "source": "web_search"
                })
            
            # Get related topics
            for topic in data.get('RelatedTopics', [])[:num_results-len(results)]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        "title": topic.get('Text', '')[:100] + '...',
                        "url": topic.get('FirstURL', ''),
                        "description": topic.get('Text', ''),
                        "content": topic.get('Text', ''),
                        "published": "",
                        "source": "web_search"
                    })
            
            return results
            
        except Exception as e:
            return []
    
    def search_medical(self, query: str) -> str:
        """Perform medical-focused web search and return formatted context"""
        # Add medical sites to the query for better results
        medical_query = f"{query} site:mayoclinic.org OR site:webmd.com OR site:healthline.com OR site:medlineplus.gov"
        results = self.search(medical_query, num_results=3)
        
        # If no results with site restriction, try general medical search
        if not results:
            medical_query = f"medical health {query}"
            results = self.search(medical_query, num_results=3)
        
        if not results:
            return "No recent web information found."
        
        context_parts = []
        for result in results:
            if result['title'] and result['description']:
                context_parts.append(f"Source: {result['title']}\n{result['description']}")
        
        return "\n\n".join(context_parts) if context_parts else "No recent web information found."

class VoiceAssistant:
    """Enhanced voice assistant with better error handling"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = None
        self.is_speaking = False
        self.current_speech_thread = None
        self.setup_components()
    
    def setup_components(self):
        """Setup TTS and microphone"""
        # Setup TTS
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', config.TTS_RATE)
            self.tts_engine.setProperty('volume', config.TTS_VOLUME)
        except Exception as e:
            st.error(f"TTS initialization error: {e}")
        
        # Setup microphone
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except Exception as e:
            st.error(f"Microphone setup error: {e}")
    
    def speak(self, text: str) -> bool:
        """Convert text to speech"""
        if not self.tts_engine:
            return False
        
        try:
            # Clean text for better speech
            clean_text = self._clean_text_for_speech(text)
            
            def speak_thread():
                self.is_speaking = True
                try:
                    self.tts_engine.say(clean_text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    st.error(f"TTS playback error: {e}")
                finally:
                    self.is_speaking = False
            
            if self.current_speech_thread and self.current_speech_thread.is_alive():
                self.stop_speaking()
            
            self.current_speech_thread = threading.Thread(target=speak_thread, daemon=True)
            self.current_speech_thread.start()
            return True
            
        except Exception as e:
            st.error(f"TTS error: {e}")
            return False
    
    def stop_speaking(self) -> bool:
        """Stop current speech"""
        if self.tts_engine and self.is_speaking:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
                return True
            except Exception as e:
                st.error(f"Error stopping TTS: {e}")
                return False
        return False
    
    def listen_once(self) -> str:
        """Listen for a single voice input with improved error handling"""
        try:
            with self.microphone as source:
                st.info("🎤 Listening... Please speak now!")
                
                # Listen for audio with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=config.SPEECH_TIMEOUT, 
                    phrase_time_limit=config.PHRASE_TIME_LIMIT
                )
                
                st.info("🔄 Processing speech...")
                
                # Try Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    # Try alternative recognition if available
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        return text
                    except:
                        pass
                    
                st.warning("🤔 Could not understand audio")
                return ""
                
        except sr.WaitTimeoutError:
            st.warning("⏱️ No speech detected within timeout period")
            return ""
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition service error: {e}")
            return ""
        except Exception as e:
            st.error(f"❌ Error during speech recognition: {e}")
            return ""
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better TTS output"""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'#{1,6}\s', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def is_available(self) -> bool:
        """Check if voice assistant is available"""
        return self.tts_engine is not None

class ResponseFormatter:
    """Handles different response formatting modes"""
    
    @staticmethod
    def format_medical_response(response: str, mode: str = "detailed") -> str:
        """Format medical response based on mode"""
        if mode == "concise":
            return ResponseFormatter._create_concise_response(response)
        elif mode == "detailed":
            return ResponseFormatter._create_detailed_response(response)
        else:
            return response
    
    @staticmethod
    def _create_concise_response(response: str) -> str:
        """Create a concise version of the response"""
        # Split into sentences and take the most important ones
        sentences = response.split('. ')
        if len(sentences) <= 3:
            return response
        
        # Keep first sentence (usually the main point) and last sentence (usually advice)
        important_sentences = [sentences[0]]
        
        # Add middle sentences that contain important keywords
        important_keywords = ['important', 'should', 'recommend', 'consult', 'seek', 'avoid', 'treatment']
        for sentence in sentences[1:-1]:
            if any(keyword in sentence.lower() for keyword in important_keywords):
                important_sentences.append(sentence)
                if len(important_sentences) >= 2:
                    break
        
        # Always include the last sentence if it's advice
        if sentences[-1] and any(word in sentences[-1].lower() for word in ['consult', 'seek', 'contact', 'visit']):
            important_sentences.append(sentences[-1])
        
        return '. '.join(important_sentences) + '.'
    
    @staticmethod
    def _create_detailed_response(response: str) -> str:
        """Create a detailed version of the response"""
        # Add formatting to make the response more structured
        sections = response.split('\n\n')
        formatted_sections = []
        
        for i, section in enumerate(sections):
            if section.strip():
                # Add section headers for better readability
                if i == 0 and not section.startswith('**'):
                    formatted_sections.append(f"**Medical Information:**\n{section}")
                else:
                    formatted_sections.append(section)
        
        return '\n\n'.join(formatted_sections)

class SessionManager:
    """Manages session state and conversation history"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'vector_db_initialized' not in st.session_state:
            st.session_state.vector_db_initialized = False
        
        if 'voice_enabled' not in st.session_state:
            st.session_state.voice_enabled = False
        
        if 'response_mode' not in st.session_state:
            st.session_state.response_mode = "detailed"
        
        if 'selected_provider' not in st.session_state:
            st.session_state.selected_provider = "groq"
        
        if 'web_search_enabled' not in st.session_state:
            st.session_state.web_search_enabled = False
    
    @staticmethod
    def add_message(role: str, content: str):
        """Add message to session state"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(message)
    
    @staticmethod
    def clear_messages():
        """Clear all messages from session state"""
        st.session_state.messages = []
    
    @staticmethod
    def export_conversation() -> str:
        """Export conversation history as formatted text"""
        if not st.session_state.messages:
            return "No conversation to export."
        
        export_text = f"Medical AI Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "=" * 50 + "\n\n"
        
        for message in st.session_state.messages:
            timestamp = datetime.fromisoformat(message.get('timestamp', datetime.now().isoformat()))
            role = message['role'].capitalize()
            content = message['content']
            
            export_text += f"[{timestamp.strftime('%H:%M:%S')}] {role}:\n{content}\n\n"
        
        return export_text

class FileHandler:
    """Handles file upload and processing"""
    
    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """Save uploaded file to temporary location"""
        try:
            # Create uploads directory if it doesn't exist
            os.makedirs(config.UPLOAD_DIR, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{uploaded_file.name}"
            file_path = os.path.join(config.UPLOAD_DIR, filename)
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return file_path
            
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return ""
    
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if not uploaded_file:
            return False, "No file uploaded"
        
        # Check file size
        if uploaded_file.size > config.MAX_FILE_SIZE:
            return False, f"File too large. Maximum size: {config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        
        # Check file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in config.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Supported: {', '.join(config.SUPPORTED_FORMATS)}"
        
        return True, "File is valid"

# Utility functions for common operations
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text.strip()

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > max_length * 0.8:
        return truncated[:last_period + 1]
    else:
        return truncated + "..."

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%H:%M:%S")
    except:
        return "Unknown"