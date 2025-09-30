import time
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from config.logging_config import get_logger

logger = get_logger(__name__)

class GenericVectorStore:
    """Generic vector store manager that works with any data type."""
    
    def __init__(self, persist_directory: str, embeddings: GoogleGenerativeAIEmbeddings, collection_name: str = "data_collection"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.vectorstore: Optional[Chroma] = None
        self.collection_name = collection_name
        
    def build_vectorstore(self, documents: List[Document], force_rebuild: bool = False) -> Chroma:
        """Build or load the vector store from documents."""
        if not force_rebuild and self._vectorstore_exists():
            logger.info("Loading existing vector store...")
            self.vectorstore = self._load_vectorstore()
        else:
            logger.info("Building new vector store...")
            self.vectorstore = self._create_vectorstore(documents)
        
        return self.vectorstore
    
    def _vectorstore_exists(self) -> bool:
        """Check if vector store already exists."""
        return (self.persist_directory / "chroma.sqlite3").exists()
    
    def _load_vectorstore(self) -> Chroma:
        """Load existing vector store."""
        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            logger.info(f"Loaded vector store with {vectorstore._collection.count()} documents")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}")
            return None
    
    def _create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create new vector store from documents."""
        if not documents:
            logger.warning("No documents provided for vector store creation")
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
            
            logger.info(f"Created vector store with {len(documents)} documents")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict[str, Any] = None) -> Any:
        """Get a retriever from the vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore() first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore() first.")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with scores."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore() first.")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if self.vectorstore is None:
            return {
                "status": "not_initialized",
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
        
        try:
            count = self.vectorstore._collection.count()
            return {
                "status": "initialized",
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            if self.vectorstore is None:
                # Try to delete from disk directly
                import shutil
                if self.persist_directory.exists():
                    shutil.rmtree(self.persist_directory)
                    logger.info(f"Deleted vector store directory: {self.persist_directory}")
                    return True
                return False
            
            # Delete using ChromaDB client
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def rebuild_vectorstore(self, documents: List[Document]) -> bool:
        """Rebuild the vector store with new documents."""
        try:
            logger.info("Rebuilding vector store...")
            
            # Delete existing collection
            self.delete_collection()
            
            # Create new vector store
            self.vectorstore = self._create_vectorstore(documents)
            
            logger.info("Vector store rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild vector store: {e}")
            return False
