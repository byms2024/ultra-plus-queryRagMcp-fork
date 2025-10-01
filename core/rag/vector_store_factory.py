#!/usr/bin/env python3
"""
Vector Store Factory for creating different types of vector stores.
Supports ChromaDB, FAISS, and other vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.schema import Document

from config.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreInterface(ABC):
    """Interface for vector store implementations."""
    
    @abstractmethod
    def build_vectorstore(self, documents: List[Document], force_rebuild: bool = False):
        """Build or load the vector store from documents."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with scores."""
        pass
    
    @abstractmethod
    def similarity_search_with_score_threshold(
        self, 
        query: str, 
        similarity_threshold: float = 0.7,
        max_results: int = 100,
        min_results: int = 1
    ) -> List[tuple]:
        """Perform similarity search with similarity threshold filtering."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        pass
    
    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        pass
    
    @abstractmethod
    def rebuild_vectorstore(self, documents: List[Document]) -> bool:
        """Rebuild the vector store with new documents."""
        pass


class ChromaVectorStore(VectorStoreInterface):
    """ChromaDB vector store implementation."""
    
    def __init__(self, persist_directory: str, embeddings, collection_name: str = "data_collection"):
        from langchain_chroma import Chroma
        import chromadb
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.vectorstore: Optional[Chroma] = None
        self.collection_name = collection_name
        self._chromadb = chromadb
    
    def build_vectorstore(self, documents: List[Document], force_rebuild: bool = False):
        """Build or load the ChromaDB vector store from documents."""
        if not force_rebuild and self._vectorstore_exists():
            logger.info("Loading existing ChromaDB vector store...")
            self.vectorstore = self._load_vectorstore()
        else:
            logger.info("Building new ChromaDB vector store...")
            self.vectorstore = self._create_vectorstore(documents)
        
        return self.vectorstore
    
    def _vectorstore_exists(self) -> bool:
        """Check if ChromaDB vector store already exists."""
        return (self.persist_directory / "chroma.sqlite3").exists()
    
    def _load_vectorstore(self):
        """Load existing ChromaDB vector store."""
        from langchain_chroma import Chroma
        
        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            logger.info(f"Loaded ChromaDB vector store with {vectorstore._collection.count()} documents")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load existing ChromaDB vector store: {e}")
            return None
    
    def _create_vectorstore(self, documents: List[Document]):
        """Create new ChromaDB vector store from documents."""
        from langchain_chroma import Chroma
        import time
        
        if not documents:
            logger.warning("No documents provided for ChromaDB vector store creation")
            return None
        
        try:
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory)
            )
            
            logger.info(f"Created ChromaDB vector store with {len(documents)} documents")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create ChromaDB vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search using ChromaDB."""
        if self.vectorstore is None:
            raise ValueError("ChromaDB vector store not initialized. Call build_vectorstore() first.")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"ChromaDB similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with scores using ChromaDB."""
        if self.vectorstore is None:
            raise ValueError("ChromaDB vector store not initialized. Call build_vectorstore() first.")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"ChromaDB similarity search with score failed: {e}")
            return []
    
    def similarity_search_with_score_threshold(
        self, 
        query: str, 
        similarity_threshold: float = 0.7,
        max_results: int = 100,
        min_results: int = 1
    ) -> List[tuple]:
        """Perform similarity search with threshold filtering using ChromaDB."""
        if self.vectorstore is None:
            raise ValueError("ChromaDB vector store not initialized. Call build_vectorstore() first.")
        
        try:
            # Get more results than needed to ensure we have enough after filtering
            search_k = max(max_results * 2, 50)  # Search for 2x max_results to account for filtering
            results = self.vectorstore.similarity_search_with_score(query, k=search_k)
            
            # Filter results by similarity threshold
            # Note: ChromaDB returns cosine similarity scores (higher = more similar)
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= similarity_threshold
            ]
            
            # Apply result limits
            if len(filtered_results) > max_results:
                filtered_results = filtered_results[:max_results]
            
            # Ensure minimum results if we have any
            if len(filtered_results) < min_results and len(results) > 0:
                # Take the best results even if they don't meet threshold
                filtered_results = results[:min_results]
            
            logger.info(f"Found {len(filtered_results)} documents above threshold {similarity_threshold} for query: {query[:50]}...")
            return filtered_results
            
        except Exception as e:
            logger.error(f"ChromaDB threshold similarity search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB vector store statistics."""
        if self.vectorstore is None:
            return {
                "status": "not_initialized",
                "type": "chroma",
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
        
        try:
            count = self.vectorstore._collection.count()
            return {
                "status": "initialized",
                "type": "chroma",
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB vector store stats: {e}")
            return {
                "status": "error",
                "type": "chroma",
                "error": str(e),
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
    
    def delete_collection(self) -> bool:
        """Delete the entire ChromaDB collection."""
        try:
            if self.vectorstore is None:
                # Try to delete from disk directly
                import shutil
                if self.persist_directory.exists():
                    shutil.rmtree(self.persist_directory)
                    logger.info(f"Deleted ChromaDB vector store directory: {self.persist_directory}")
                    return True
                return False
            
            # Delete using ChromaDB client
            client = self._chromadb.PersistentClient(path=str(self.persist_directory))
            client.delete_collection(self.collection_name)
            logger.info(f"Deleted ChromaDB collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB collection: {e}")
            return False
    
    def rebuild_vectorstore(self, documents: List[Document]) -> bool:
        """Rebuild the ChromaDB vector store with new documents."""
        try:
            logger.info("Rebuilding ChromaDB vector store...")
            
            # Delete existing collection
            self.delete_collection()
            
            # Create new vector store
            self.vectorstore = self._create_vectorstore(documents)
            
            logger.info("ChromaDB vector store rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild ChromaDB vector store: {e}")
            return False


class FAISSVectorStore(VectorStoreInterface):
    """FAISS vector store implementation."""
    
    def __init__(self, persist_directory: str, embeddings, collection_name: str = "data_collection"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.vectorstore = None
        self.collection_name = collection_name
        self.index_path = self.persist_directory / f"{collection_name}.faiss"
        self.pkl_path = self.persist_directory / f"{collection_name}.pkl"
    
    def build_vectorstore(self, documents: List[Document], force_rebuild: bool = False):
        """Build or load the FAISS vector store from documents."""
        if not force_rebuild and self._vectorstore_exists():
            logger.info("Loading existing FAISS vector store...")
            self.vectorstore = self._load_vectorstore()
        else:
            logger.info("Building new FAISS vector store...")
            self.vectorstore = self._create_vectorstore(documents)
        
        return self.vectorstore
    
    def _vectorstore_exists(self) -> bool:
        """Check if FAISS vector store already exists."""
        return self.index_path.exists() and self.pkl_path.exists()
    
    def _load_vectorstore(self):
        """Load existing FAISS vector store."""
        from langchain_community.vectorstores import FAISS
        
        try:
            vectorstore = FAISS.load_local(
                str(self.persist_directory),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded FAISS vector store with {vectorstore.index.ntotal} documents")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load existing FAISS vector store: {e}")
            return None
    
    def _create_vectorstore(self, documents: List[Document]):
        """Create new FAISS vector store from documents."""
        from langchain_community.vectorstores import FAISS
        
        if not documents:
            logger.warning("No documents provided for FAISS vector store creation")
            return None
        
        try:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Save to disk
            vectorstore.save_local(str(self.persist_directory))
            
            logger.info(f"Created FAISS vector store with {len(documents)} documents")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search using FAISS."""
        if self.vectorstore is None:
            raise ValueError("FAISS vector store not initialized. Call build_vectorstore() first.")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"FAISS similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with scores using FAISS."""
        if self.vectorstore is None:
            raise ValueError("FAISS vector store not initialized. Call build_vectorstore() first.")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"FAISS similarity search with score failed: {e}")
            return []
    
    def similarity_search_with_score_threshold(
        self, 
        query: str, 
        similarity_threshold: float = 0.7,
        max_results: int = 100,
        min_results: int = 1
    ) -> List[tuple]:
        """Perform similarity search with threshold filtering using FAISS."""
        if self.vectorstore is None:
            raise ValueError("FAISS vector store not initialized. Call build_vectorstore() first.")
        
        try:
            # Get more results than needed to ensure we have enough after filtering
            search_k = max(max_results * 2, 50)  # Search for 2x max_results to account for filtering
            results = self.vectorstore.similarity_search_with_score(query, k=search_k)
            
            # Filter results by similarity threshold
            # Note: FAISS typically returns distance scores (lower = more similar)
            # We need to convert distance to similarity score (1 - distance) for threshold comparison
            filtered_results = []
            for doc, distance in results:
                # Convert distance to similarity (assuming normalized distance 0-1)
                # For FAISS, lower distance = higher similarity
                similarity = max(0.0, 1.0 - distance)
                if similarity >= similarity_threshold:
                    filtered_results.append((doc, similarity))
            
            # Apply result limits
            if len(filtered_results) > max_results:
                filtered_results = filtered_results[:max_results]
            
            # Ensure minimum results if we have any
            if len(filtered_results) < min_results and len(results) > 0:
                # Take the best results even if they don't meet threshold
                # Convert distance back to similarity for consistency
                best_results = []
                for doc, distance in results[:min_results]:
                    similarity = max(0.0, 1.0 - distance)
                    best_results.append((doc, similarity))
                filtered_results = best_results
            
            logger.info(f"Found {len(filtered_results)} documents above threshold {similarity_threshold} for query: {query[:50]}...")
            return filtered_results
            
        except Exception as e:
            logger.error(f"FAISS threshold similarity search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS vector store statistics."""
        if self.vectorstore is None:
            return {
                "status": "not_initialized",
                "type": "faiss",
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
        
        try:
            count = self.vectorstore.index.ntotal
            return {
                "status": "initialized",
                "type": "faiss",
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Failed to get FAISS vector store stats: {e}")
            return {
                "status": "error",
                "type": "faiss",
                "error": str(e),
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
    
    def delete_collection(self) -> bool:
        """Delete the entire FAISS collection."""
        try:
            import shutil
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
                logger.info(f"Deleted FAISS vector store directory: {self.persist_directory}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete FAISS collection: {e}")
            return False
    
    def rebuild_vectorstore(self, documents: List[Document]) -> bool:
        """Rebuild the FAISS vector store with new documents."""
        try:
            logger.info("Rebuilding FAISS vector store...")
            
            # Delete existing collection
            self.delete_collection()
            
            # Create new vector store
            self.vectorstore = self._create_vectorstore(documents)
            
            logger.info("FAISS vector store rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS vector store: {e}")
            return False


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create(
        store_type: str, 
        persist_directory: str, 
        embeddings, 
        collection_name: str = "data_collection"
    ) -> VectorStoreInterface:
        """
        Create a vector store instance based on the store type.
        
        Args:
            store_type: Type of vector store ("chroma", "faiss")
            persist_directory: Directory to persist the vector store
            embeddings: Embeddings instance
            collection_name: Name of the collection/index
            
        Returns:
            VectorStoreInterface: Appropriate vector store implementation
            
        Raises:
            ValueError: If store_type is not supported
        """
        store_type = store_type.lower()
        
        if store_type == "chroma":
            return ChromaVectorStore(persist_directory, embeddings, collection_name)
        elif store_type == "faiss":
            return FAISSVectorStore(persist_directory, embeddings, collection_name)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}. Supported types: chroma, faiss")
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported vector store types."""
        return ["chroma", "faiss"]
    
    @staticmethod
    def get_default_type() -> str:
        """Get the default vector store type."""
        return "chroma"
