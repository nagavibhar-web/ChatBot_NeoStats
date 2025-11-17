import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

@dataclass
class Config:
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY","AIzaSyB8D_MkjWH8hxf8XpXJe6zV4s11w0s3-OE")  # Get FREE key from https://aistudio.google.com/
    BRAVE_API_KEY: Optional[str] = os.getenv("BRAVE_API_KEY")  # For web search
    
    # LLM Settings - Updated with multiple providers
    LLM_PROVIDERS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "groq": {
            "name": "Groq (Llama 3.1)",
            "model": "llama-3.1-8b-instant",
            "api_key_env": "GROQ_API_KEY",
            "max_tokens": 1024,
            "temperature": 0.7,
            "supports_streaming": True,
            "cost_per_1k_tokens": 0.0001,  # Very low cost
            "speed": "Very Fast"
        },
        "gemini": {
            "name": "Google Gemini",
            "model": "gemini-2.0-flash",
            "api_key_env": "GOOGLE_API_KEY", 
            "max_tokens": 2048,
            "temperature": 0.7,
            "supports_streaming": True,
            "cost_per_1k_tokens": 0.0,  # Free tier
            "speed": "Fast"
        }
    })
    
    # Default LLM Settings (fallback)
    DEFAULT_PROVIDER: str = "groq"
    DEFAULT_MODEL: str = "llama-3.1-8b-instant"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1024
    TOP_P: float = 1.0
    
    # Gemini Specific Settings
    GEMINI_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "gemini-pro": "gemini-pro",
        "gemini-pro-vision": "gemini-pro-vision",  # For future image support
    })
    
    # Groq Specific Settings  
    GROQ_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "llama-3.1-8b": "llama-3.1-8b-instant",
        "llama-3.1-70b": "llama-3.1-70b-versatile",
        "mixtral-8x7b": "mixtral-8x7b-32768",
        "gemma-7b": "gemma-7b-it"
    })
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # RAG Settings
    VECTOR_DB_PATH: str = "medical_vectordb11.pkl"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Voice Settings
    TTS_RATE: int = 150
    TTS_VOLUME: float = 0.9
    SPEECH_TIMEOUT: int = 5
    PHRASE_TIME_LIMIT: int = 10
    
    # Web Search Settings
    SEARCH_ENABLED: bool = True
    MAX_SEARCH_RESULTS: int = 3
    SEARCH_TIMEOUT: int = 10
    
    # Response Settings - Updated with provider-specific configs
    RESPONSE_MODES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "concise": {
            "groq": {"max_tokens": 512, "temperature": 0.5},
            "gemini": {"max_tokens": 512, "temperature": 0.5},
            "description": "Short, summarized replies"
        },
        "detailed": {
            "groq": {"max_tokens": 1536, "temperature": 0.7},
            "gemini": {"max_tokens": 2048, "temperature": 0.7},
            "description": "Expanded, in-depth responses"
        }
    })
    
    # File Settings
    UPLOAD_DIR: str = "uploads"
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".docx"])
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # System Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers based on API keys"""
        available = []
        
        if self.GROQ_API_KEY and self.GROQ_API_KEY != "":
            available.append("groq")
            
        if self.GOOGLE_API_KEY and self.GOOGLE_API_KEY != "":
            available.append("gemini")
            
        return available
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get configuration info for a specific provider"""
        return self.LLM_PROVIDERS.get(provider, {})
    
    def get_provider_models(self, provider: str) -> Dict[str, str]:
        """Get available models for a provider"""
        if provider == "groq":
            return self.GROQ_MODELS
        elif provider == "gemini":
            return self.GEMINI_MODELS
        return {}

# Create config instance
config = Config()

# Ensure upload directory exists
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
