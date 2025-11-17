from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import streamlit as st # type: ignore
from groq import Groq # type: ignore
import openai # type: ignore
import google.generativeai as genai # type: ignore
from config.config import config
import logging
import time
from functools import wraps

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function calls on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            raise last_exception
        return wrapper
    return decorator

class BaseLLM(ABC):
    """Abstract base class for LLM models"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass

class GroqLLM(BaseLLM):
    """Groq LLM implementation"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.DEFAULT_MODEL
        self.client = None
        self.provider_info = config.get_provider_info("groq")
        
        if self.api_key and self.api_key.strip():
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                if config.DEBUG:
                    st.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("Groq API key not provided")
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None and self.api_key.strip() != ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Groq model information"""
        return {
            "provider": "groq",
            "model": self.model,
            "available_models": list(config.GROQ_MODELS.keys()),
            "supports_streaming": self.provider_info.get("supports_streaming", True),
            "cost_per_1k_tokens": self.provider_info.get("cost_per_1k_tokens", 0.0001),
            "speed": self.provider_info.get("speed", "Very Fast")
        }
    
    @retry_on_failure(max_retries=2, delay=0.5)
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response using Groq API"""
        if not self.is_available():
            error_msg = "Groq client not available. Please check your API key."
            logger.error(error_msg)
            return error_msg
        
        try:
            # Validate messages format
            if not messages or not isinstance(messages, list):
                raise ValueError("Messages must be a non-empty list")
            
            # Extract parameters with defaults
            temperature = kwargs.get('temperature', config.TEMPERATURE)
            max_tokens = kwargs.get('max_tokens', config.MAX_TOKENS)
            model = kwargs.get('model', self.model)
            
            # Validate model
            if model not in config.GROQ_MODELS.values():
                logger.warning(f"Model {model} not in supported models. Using default.")
                model = self.model
            
            # Clamp values to acceptable ranges
            temperature = max(0.0, min(2.0, temperature))
            max_tokens = max(1, min(32768, max_tokens))
            
            logger.debug(f"Generating response with Groq - Model: {model}, Temp: {temperature}, Max tokens: {max_tokens}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=config.TOP_P,
                stream=False
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    logger.info(f"Successfully generated response with {len(content)} characters")
                    return content.strip()
                else:
                    return "Empty response received from Groq API."
            else:
                return "No response choices returned from Groq API."
                
        except Exception as e:
            error_msg = f"Error generating response with Groq: {str(e)}"
            logger.error(error_msg)
            return error_msg

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model
        
        if self.api_key and self.api_key.strip():
            try:
                openai.api_key = self.api_key
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OpenAI API key not provided")
    
    def is_available(self) -> bool:
        return self.api_key is not None and self.api_key.strip() != ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        return {
            "provider": "openai",
            "model": self.model,
            "available_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            "supports_streaming": True,
            "cost_per_1k_tokens": 0.002,
            "speed": "Medium"
        }
    
    @retry_on_failure(max_retries=2, delay=1.0)
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response using OpenAI API"""
        if not self.is_available():
            error_msg = "OpenAI client not available. Please check your API key."
            logger.error(error_msg)
            return error_msg
        
        try:
            # Validate messages format
            if not messages or not isinstance(messages, list):
                raise ValueError("Messages must be a non-empty list")
            
            temperature = kwargs.get('temperature', config.TEMPERATURE)
            max_tokens = kwargs.get('max_tokens', config.MAX_TOKENS)
            model = kwargs.get('model', self.model)
            
            # Clamp values to acceptable ranges
            temperature = max(0.0, min(2.0, temperature))
            max_tokens = max(1, min(4096, max_tokens))
            
            logger.debug(f"Generating response with OpenAI - Model: {model}, Temp: {temperature}, Max tokens: {max_tokens}")
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    logger.info(f"Successfully generated response with {len(content)} characters")
                    return content.strip()
                else:
                    return "Empty response received from OpenAI API."
            else:
                return "No response choices returned from OpenAI API."
                
        except Exception as e:
            error_msg = f"Error generating response with OpenAI: {str(e)}"
            logger.error(error_msg)
            return error_msg

class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.model = model
        self.client = None
        self.provider_info = config.get_provider_info("gemini")
        
        if self.api_key and self.api_key.strip():
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info("Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                if config.DEBUG:
                    st.error(f"Failed to initialize Gemini client: {e}")
                self.client = None
        else:
            logger.warning("Gemini API key not provided")
            self.client = None
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None and self.api_key.strip() != ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information"""
        return {
            "provider": "gemini",
            "model": self.model,
            "available_models": list(config.GEMINI_MODELS.keys()),
            "supports_streaming": self.provider_info.get("supports_streaming", True),
            "cost_per_1k_tokens": self.provider_info.get("cost_per_1k_tokens", 0.0),
            "speed": self.provider_info.get("speed", "Fast")
        }
    
    @retry_on_failure(max_retries=2, delay=1.0)
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response using Gemini API"""
        if not self.is_available():
            error_msg = "Gemini client not available. Please check your API key."
            logger.error(error_msg)
            return error_msg
        
        try:
            # Validate messages format
            if not messages or not isinstance(messages, list):
                raise ValueError("Messages must be a non-empty list")
            
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_prompt(messages)
            if not prompt.strip():
                return "Empty prompt generated from messages."
            
            temperature = kwargs.get('temperature', config.TEMPERATURE)
            max_tokens = kwargs.get('max_tokens', config.MAX_TOKENS)
            
            # Clamp values to acceptable ranges for Gemini
            temperature = max(0.0, min(1.0, temperature))
            max_tokens = max(1, min(8192, max_tokens))
            
            logger.debug(f"Generating response with Gemini - Model: {self.model}, Temp: {temperature}, Max tokens: {max_tokens}")
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )
            
            if response.text:
                logger.info(f"Successfully generated response with {len(response.text)} characters")
                return response.text.strip()
            else:
                # Check if response was blocked
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    return f"Response blocked by safety filters: {response.prompt_feedback.block_reason}"
                return "Empty response received from Gemini API."
                
        except Exception as e:
            error_msg = f"Error generating response with Gemini: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to Gemini prompt format"""
        prompt_parts = []
        
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            role = message.get('role', '').lower()
            content = message.get('content', '').strip()
            
            if not content:
                continue
            
            if role == 'system':
                prompt_parts.append(f"System Instructions: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)

class LLMManager:
    """Manager class to handle multiple LLM providers"""
    
    def __init__(self):
        self.providers = {
            'groq': GroqLLM(),
            'openai': OpenAILLM(),
            'gemini': GeminiLLM()
        }
        self.default_provider = config.DEFAULT_PROVIDER
        logger.info(f"LLM Manager initialized with default provider: {self.default_provider}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        available = []
        for name, provider in self.providers.items():
            if provider.is_available():
                available.append(name)
                logger.debug(f"Provider {name} is available")
            else:
                logger.debug(f"Provider {name} is not available")
        
        logger.info(f"Available providers: {available}")
        return available
    
    def get_provider(self, provider_name: str = None) -> BaseLLM:
        """Get LLM provider by name"""
        if provider_name is None:
            provider_name = self.default_provider
        
        if provider_name not in self.providers:
            logger.warning(f"Provider {provider_name} not found. Using default: {self.default_provider}")
            if config.DEBUG:
                st.warning(f"Provider {provider_name} not found. Using default.")
            provider_name = self.default_provider
        
        return self.providers[provider_name]
    
    def get_provider_info(self, provider_name: str = None) -> Dict[str, Any]:
        """Get information about a specific provider"""
        provider = self.get_provider(provider_name)
        return provider.get_model_info()
    
    def get_all_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available providers"""
        info = {}
        available_providers = self.get_available_providers()
        
        for provider_name in available_providers:
            try:
                info[provider_name] = self.get_provider_info(provider_name)
            except Exception as e:
                logger.error(f"Error getting info for provider {provider_name}: {e}")
                info[provider_name] = {"error": str(e)}
        
        return info
    
    def switch_provider(self, new_provider: str) -> bool:
        """Switch the default provider"""
        if new_provider in self.providers:
            provider = self.providers[new_provider]
            if provider.is_available():
                self.default_provider = new_provider
                logger.info(f"Switched default provider to: {new_provider}")
                return True
            else:
                logger.warning(f"Cannot switch to {new_provider}: provider not available")
                return False
        else:
            logger.error(f"Provider {new_provider} not found")
            return False
    
    def generate_response(self, messages: List[Dict], provider: str = None, **kwargs) -> str:
        """Generate response using specified provider"""
        # Validate input
        if not messages:
            return "No messages provided for response generation."
        
        llm = self.get_provider(provider)
        
        if not llm.is_available():
            # Try to find an available provider
            available_providers = self.get_available_providers()
            if available_providers:
                fallback_provider = available_providers[0]
                llm = self.get_provider(fallback_provider)
                logger.info(f"Using {fallback_provider} as fallback provider")
                if config.DEBUG:
                    st.info(f"Using {fallback_provider} as fallback provider.")
            else:
                error_msg = "No LLM providers available. Please check your API keys."
                logger.error(error_msg)
                return error_msg
        
        # Log the request
        provider_info = llm.get_model_info()
        logger.info(f"Generating response using {provider_info.get('provider', 'unknown')} with model {provider_info.get('model', 'unknown')}")
        
        start_time = time.time()
        response = llm.generate_response(messages, **kwargs)
        end_time = time.time()
        
        logger.info(f"Response generated in {end_time - start_time:.2f} seconds")
        
        return response

# Global LLM manager instance
llm_manager = LLMManager()
