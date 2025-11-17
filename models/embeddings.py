import numpy as np # type: ignore
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from config.config import config
import streamlit as st # type: ignore

class EmbeddingModel:
    """Handles text embeddings for RAG"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            if config.DEBUG:
                st.success(f"✅ Embedding model '{self.model_name}' loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load embedding model: {e}")
            self.model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        if not self.model:
            raise ValueError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            st.error(f"Error encoding texts: {e}")
            return np.array([])
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        return self.encode([text])[0] if self.model else np.array([])
    
    def is_available(self) -> bool:
        """Check if model is available"""
        return self.model is not None

class VectorDatabase:
    """Enhanced vector database with metadata support"""
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chunks: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self._index_built = False
    
    def add_documents(self, 
                     texts: List[str], 
                     metadata: List[Dict[str, Any]] = None,
                     source: str = "unknown") -> bool:
        """Add documents to the vector database"""
        if not self.embedding_model.is_available():
            st.error("Embedding model not available")
            return False
        
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Prepare metadata
            if metadata is None:
                metadata = [{"source": source, "chunk_id": i + len(self.chunks)} 
                          for i in range(len(texts))]
            
            # Add to database
            self.chunks.extend(texts)
            self.embeddings.extend(embeddings)
            self.metadata.extend(metadata)
            
            self._index_built = True
            
            if config.DEBUG:
                st.success(f"✅ Added {len(texts)} documents to vector database")
            
            return True
            
        except Exception as e:
            st.error(f"Error adding documents: {e}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         k: int = None,
                         similarity_threshold: float = None,
                         filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents with optional filtering"""
        
        if not self._index_built or not self.embeddings:
            return []
        
        k = k or config.TOP_K_RESULTS
        similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode_single(query)
            if query_embedding.size == 0:
                return []
            
            # Calculate similarities
            embeddings_array = np.array(self.embeddings)
            similarities = cosine_similarity([query_embedding], embeddings_array)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                similarity_score = float(similarities[idx])
                
                # Apply similarity threshold
                if similarity_score < similarity_threshold:
                    continue
                
                result = {
                    "content": self.chunks[idx],
                    "similarity": similarity_score,
                    "metadata": self.metadata[idx],
                    "index": int(idx)
                }
                
                # Apply metadata filtering
                if filter_metadata:
                    metadata_match = all(
                        self.metadata[idx].get(key) == value 
                        for key, value in filter_metadata.items()
                    )
                    if not metadata_match:
                        continue
                
                results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Error in similarity search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_chunks": len(self.chunks),
            "total_embeddings": len(self.embeddings),
            "embedding_dimension": self.embeddings[0].shape[0] if self.embeddings else 0,
            "sources": list(set(meta.get("source", "unknown") for meta in self.metadata)),
            "index_built": self._index_built
        }
    
    def clear(self):
        """Clear all data from the database"""
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        self._index_built = False
        
        if config.DEBUG:
            st.info("🗑️ Vector database cleared")
    
    def save_to_file(self, filepath: str = None) -> bool:
        """Save vector database to file"""
        filepath = filepath or config.VECTOR_DB_PATH
        
        try:
            data = {
                "chunks": self.chunks,
                "embeddings": self.embeddings,
                "metadata": self.metadata,
                "model_name": self.embedding_model.model_name,
                "index_built": self._index_built
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            if config.DEBUG:
                st.success(f"✅ Vector database saved to {filepath}")
            
            return True
            
        except Exception as e:
            st.error(f"Error saving vector database: {e}")
            return False
    
    def load_from_file(self, filepath: str = None) -> bool:
        """Load vector database from file"""
        filepath = filepath or config.VECTOR_DB_PATH
        
        if not os.path.exists(filepath):
            if config.DEBUG:
                st.warning(f"Vector database file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data.get("chunks", [])
            self.embeddings = data.get("embeddings", [])
            self.metadata = data.get("metadata", [])
            self._index_built = data.get("index_built", False)
            
            # Verify model compatibility
            saved_model = data.get("model_name", "")
            if saved_model and saved_model != self.embedding_model.model_name:
                st.warning(f"Model mismatch: saved={saved_model}, current={self.embedding_model.model_name}")
            
            if config.DEBUG:
                st.success(f"✅ Vector database loaded from {filepath}")
                st.info(f"📊 Loaded {len(self.chunks)} chunks")
            
            return True
            
        except Exception as e:
            st.error(f"Error loading vector database: {e}")
            return False

class RAGRetriever:
    """High-level RAG retriever combining search and context generation"""
    
    def __init__(self, vector_db: VectorDatabase = None):
        self.vector_db = vector_db or VectorDatabase()
    
    def retrieve_context(self, 
                        query: str, 
                        max_context_length: int = 2000,
                        include_metadata: bool = True) -> Dict[str, Any]:
        """Retrieve relevant context for a query"""
        
        results = self.vector_db.similarity_search(query)
        
        if not results:
            return {
                "context": "No relevant context found in the knowledge base.",
                "sources": [],
                "num_chunks": 0,
                "avg_similarity": 0.0
            }
        
        # Combine context from results
        context_parts = []
        sources = set()
        similarities = []
        
        current_length = 0
        for result in results:
            content = result["content"]
            
            # Check if adding this content would exceed max length
            if current_length + len(content) > max_context_length:
                # Truncate content to fit
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only add if substantial content fits
                    content = content[:remaining_space] + "..."
                    context_parts.append(content)
                break
            
            context_parts.append(content)
            current_length += len(content)
            
            # Track sources and similarities
            source = result["metadata"].get("source", "unknown")
            sources.add(source)
            similarities.append(result["similarity"])
        
        # Combine context
        context = "\n\n".join(context_parts)
        
        return {
            "context": context,
            "sources": list(sources),
            "num_chunks": len(context_parts),
            "avg_similarity": np.mean(similarities) if similarities else 0.0,
            "results": results[:len(context_parts)]  # Return the used results
        }
    
    def add_documents_from_texts(self, texts: List[str], source: str = "uploaded") -> bool:
        """Add documents from text list"""
        return self.vector_db.add_documents(texts, source=source)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get retriever and database statistics"""
        db_stats = self.vector_db.get_stats()
        return {
            **db_stats,
            "retriever_ready": self.vector_db._index_built and len(self.vector_db.chunks) > 0
        }