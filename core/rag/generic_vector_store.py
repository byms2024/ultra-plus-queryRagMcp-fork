from typing import List, Dict, Any
from pathlib import Path
from langchain.schema import Document

from .vector_store_factory import VectorStoreFactory, VectorStoreInterface
from config.logging_config import get_logger

logger = get_logger(__name__)

class GenericVectorStore:
    """Generic vector store manager that works with any data type and vector store backend."""
    
    def __init__(self, persist_directory: str, embeddings, collection_name: str = "data_collection", vector_store_type: str = "chroma"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.vector_store_type = vector_store_type
        
        # Create the appropriate vector store implementation
        self.vectorstore_impl: VectorStoreInterface = VectorStoreFactory.create(
            store_type=vector_store_type,
            persist_directory=str(self.persist_directory),
            embeddings=embeddings,
            collection_name=collection_name
        )
        
    def build_vectorstore(self, documents: List[Document], force_rebuild: bool = False):
        """Build or load the vector store from documents."""
        return self.vectorstore_impl.build_vectorstore(documents, force_rebuild)
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict[str, Any] = None) -> Any:
        """Get a retriever from the vector store."""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        # Get the underlying vectorstore instance for retriever creation
        vectorstore = self.vectorstore_impl.build_vectorstore([], force_rebuild=False)
        if vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore() first.")
        
        return vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        return self.vectorstore_impl.similarity_search(query, k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with scores."""
        return self.vectorstore_impl.similarity_search_with_score(query, k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = self.vectorstore_impl.get_stats()
        stats["vector_store_type"] = self.vector_store_type
        return stats
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        return self.vectorstore_impl.delete_collection()
    
    def rebuild_vectorstore(self, documents: List[Document]) -> bool:
        """Rebuild the vector store with new documents."""
        return self.vectorstore_impl.rebuild_vectorstore(documents)
