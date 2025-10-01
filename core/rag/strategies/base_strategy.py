"""
Base Retrieval Strategy Interface.

This module defines the abstract base class for all RAG retrieval strategies.
It implements the Strategy Pattern to provide flexible retrieval approaches
while maintaining a consistent interface.

Strategy Pattern Benefits:
- Encapsulates algorithms: Each strategy encapsulates a specific retrieval algorithm
- Makes algorithms interchangeable: Can switch strategies at runtime
- Reduces conditional logic: Eliminates complex if/else chains in client code
- Easy to extend: Add new strategies without modifying existing code
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.rag.generic_vector_store import GenericVectorStore
    from core.rag.config.rag_config import RAGConfig


class BaseRetrievalStrategy(ABC):
    """
    Abstract base class for RAG retrieval strategies.
    
    This class defines the interface that all retrieval strategies must implement.
    It follows the Strategy Pattern to provide a consistent interface while
    allowing different retrieval algorithms to be used interchangeably.
    
    Strategy Pattern Implementation:
    - Context: GenericRAGAgent (uses strategies)
    - Strategy: Concrete implementations (TopKStrategy, HybridStrategy)
    - Concrete Strategy: Specific algorithms for retrieval
    
    Example:
        class MyCustomStrategy(BaseRetrievalStrategy):
            def search_relevant_chunks(self, query, vectorstore, config):
                # Implement custom retrieval logic
                return results
    """
    
    @abstractmethod
    def search_relevant_chunks(
        self, 
        query: str, 
        vectorstore: 'GenericVectorStore', 
        config: 'RAGConfig'
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using this strategy.
        
        This is the main method that all concrete strategies must implement.
        It defines the standard interface for retrieval operations.
        
        Args:
            query: User query string to search for
            vectorstore: Vector store instance containing document embeddings
            config: RAG configuration object with strategy parameters
            
        Returns:
            List of result dictionaries, each containing:
            - content: Document content (str)
            - similarity: Similarity score (float, 0.0-1.0)
            - metadata: Document metadata (dict, optional)
            
        Raises:
            NotImplementedError: If not implemented by concrete strategy
            
        Example:
            >>> strategy = TopKStrategy()
            >>> results = strategy.search_relevant_chunks(
            ...     query="Samsung refrigerators",
            ...     vectorstore=vectorstore,
            ...     config=rag_config
            ... )
            >>> print(f"Found {len(results)} results")
            >>> for result in results:
            ...     print(f"Score: {result['similarity']}, Content: {result['content'][:50]}...")
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this strategy.
        
        Returns:
            String name of the strategy (e.g., "top_k", "hybrid")
            
        Example:
            >>> strategy = TopKStrategy()
            >>> print(strategy.get_strategy_name())
            "top_k"
        """
        pass
    
    @abstractmethod
    def get_strategy_description(self) -> str:
        """
        Get a human-readable description of this strategy.
        
        Returns:
            String description explaining how the strategy works
            
        Example:
            >>> strategy = HybridStrategy()
            >>> print(strategy.get_strategy_description())
            "Hybrid strategy: Quality-first with intelligent fallback mechanisms"
        """
        pass
    
    def validate_config(self, config: 'RAGConfig') -> None:
        """
        Validate that the configuration is compatible with this strategy.
        
        This method provides a hook for strategies to validate their specific
        configuration requirements. By default, it validates the general
        configuration, but strategies can override this for additional checks.
        
        Args:
            config: RAG configuration to validate
            
        Raises:
            ValueError: If configuration is invalid for this strategy
            
        Example:
            >>> strategy = TopKStrategy()
            >>> config = RAGConfig(retrieval_strategy="top_k", top_k=50)
            >>> strategy.validate_config(config)  # Should pass
            
            >>> bad_config = RAGConfig(top_k=-1)
            >>> strategy.validate_config(bad_config)  # Should raise ValueError
        """
        # Default validation - strategies can override for additional checks
        config.validate()
    
    def get_performance_characteristics(self) -> Dict[str, Any]:
        """
        Get performance characteristics of this strategy.
        
        Returns:
            Dictionary containing performance metrics and characteristics:
            - latency: Expected latency category ("low", "medium", "high")
            - memory_usage: Expected memory usage ("low", "medium", "high")
            - quality: Expected result quality ("variable", "high", "medium")
            - predictability: Result predictability ("high", "medium", "low")
            - scalability: Scalability characteristics ("good", "moderate", "limited")
            
        Example:
            >>> strategy = TopKStrategy()
            >>> perf = strategy.get_performance_characteristics()
            >>> print(f"Latency: {perf['latency']}")
            >>> print(f"Quality: {perf['quality']}")
        """
        return {
            "latency": "medium",
            "memory_usage": "medium", 
            "quality": "variable",
            "predictability": "medium",
            "scalability": "good"
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}({self.get_strategy_name()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return f"{self.__class__.__name__}(name='{self.get_strategy_name()}', description='{self.get_strategy_description()}')"
