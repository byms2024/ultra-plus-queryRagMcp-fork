"""
Top-K Retrieval Strategy Implementation.

This module implements the traditional top-k similarity search strategy.
It returns a fixed number of results based on similarity scores, without
quality filtering.

Strategy Characteristics:
- Predictable: Always returns exactly K results
- Fast: Single vector search operation
- Simple: Straightforward implementation
- Quality: Variable (may include low-similarity results)
- Best For: General queries where quantity is important
"""

from typing import List, Dict, Any, TYPE_CHECKING

from .base_strategy import BaseRetrievalStrategy

if TYPE_CHECKING:
    from core.rag.generic_vector_store import GenericVectorStore
    from core.rag.config.rag_config import RAGConfig


class TopKStrategy(BaseRetrievalStrategy):
    """
    Traditional top-k similarity search strategy.
    
    This strategy performs a standard similarity search and returns the top K
    results based on similarity scores. It does not apply any quality filtering
    or threshold-based selection.
    
    Algorithm:
    1. Perform vector similarity search with k=top_k
    2. Return top K results regardless of similarity score
    3. Convert results to standard format
    
    Performance Characteristics:
    - Latency: Low (~50ms for typical datasets)
    - Memory Usage: Low (only stores K results)
    - Quality: Variable (depends on query relevance)
    - Predictability: High (always returns K results)
    - Scalability: Good (linear with dataset size)
    
    Example:
        >>> strategy = TopKStrategy()
        >>> config = RAGConfig(retrieval_strategy="top_k", top_k=50)
        >>> results = strategy.search_relevant_chunks(
        ...     query="Samsung refrigerators",
        ...     vectorstore=vectorstore,
        ...     config=config
        ... )
        >>> print(f"Found exactly {len(results)} results")
    """
    
    def search_relevant_chunks(
        self, 
        query: str, 
        vectorstore: 'GenericVectorStore', 
        config: 'RAGConfig'
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using top-k strategy.
        
        This method implements the core top-k retrieval algorithm:
        1. Performs a similarity search with the configured top_k value
        2. Returns the top K results without quality filtering
        3. Converts results to the standard format
        
        Args:
            query: User query string to search for
            vectorstore: Vector store instance containing document embeddings
            config: RAG configuration with top_k parameter
            
        Returns:
            List of exactly top_k result dictionaries, each containing:
            - content: Document content (str)
            - similarity: Similarity score (float, 0.0-1.0)
            - metadata: Document metadata (dict, optional)
            
        Example:
            >>> strategy = TopKStrategy()
            >>> config = RAGConfig(top_k=10)
            >>> results = strategy.search_relevant_chunks(
            ...     "Samsung refrigerators", vectorstore, config
            ... )
            >>> assert len(results) == 10  # Always returns exactly top_k results
            >>> for result in results:
            ...     print(f"Score: {result['similarity']:.3f}")
        """
        # Step 1: Perform similarity search with top_k parameter
        # This gets the top K most similar documents
        search_results = vectorstore.similarity_search_with_score(
            query, k=config.top_k
        )
        
        # Step 2: Convert results to standard format
        # Each result is a tuple of (Document, similarity_score)
        formatted_results = []
        for doc, score in search_results:
            formatted_results.append({
                "content": doc.page_content,
                "similarity": float(score),  # Ensure score is a float
                "metadata": getattr(doc, 'metadata', {})
            })
        
        return formatted_results
    
    def get_strategy_name(self) -> str:
        """
        Get the name of this strategy.
        
        Returns:
            String name "top_k"
            
        Example:
            >>> strategy = TopKStrategy()
            >>> print(strategy.get_strategy_name())
            "top_k"
        """
        return "top_k"
    
    def get_strategy_description(self) -> str:
        """
        Get a human-readable description of this strategy.
        
        Returns:
            String description explaining the top-k approach
            
        Example:
            >>> strategy = TopKStrategy()
            >>> print(strategy.get_strategy_description())
            "Top-K strategy: Returns exactly K results based on similarity, no quality filtering"
        """
        return "Top-K strategy: Returns exactly K results based on similarity, no quality filtering"
    
    def get_performance_characteristics(self) -> Dict[str, Any]:
        """
        Get performance characteristics specific to top-k strategy.
        
        Returns:
            Dictionary containing top-k specific performance metrics
            
        Example:
            >>> strategy = TopKStrategy()
            >>> perf = strategy.get_performance_characteristics()
            >>> print(f"Latency: {perf['latency']}")
            >>> print(f"Predictability: {perf['predictability']}")
        """
        return {
            "latency": "low",           # Fast single search operation
            "memory_usage": "low",      # Only stores K results
            "quality": "variable",      # No quality filtering
            "predictability": "high",   # Always returns K results
            "scalability": "good"       # Linear with dataset size
        }
    
    def validate_config(self, config: 'RAGConfig') -> None:
        """
        Validate that the configuration is compatible with top-k strategy.
        
        For top-k strategy, we primarily need to validate that top_k is
        within reasonable bounds and that the strategy is set correctly.
        
        Args:
            config: RAG configuration to validate
            
        Raises:
            ValueError: If configuration is invalid for top-k strategy
            
        Example:
            >>> strategy = TopKStrategy()
            >>> config = RAGConfig(retrieval_strategy="top_k", top_k=50)
            >>> strategy.validate_config(config)  # Should pass
            
            >>> bad_config = RAGConfig(top_k=0)
            >>> strategy.validate_config(bad_config)  # Should raise ValueError
        """
        # Call base validation first
        super().validate_config(config)
        
        # Additional top-k specific validation
        if config.top_k <= 0:
            raise ValueError(f"top_k must be positive for top_k strategy, got {config.top_k}")
        
        if config.top_k > 1000:
            raise ValueError(f"top_k should not exceed 1000 for performance reasons, got {config.top_k}")
    
    def get_expected_result_count(self, config: 'RAGConfig') -> int:
        """
        Get the expected number of results for this strategy.
        
        For top-k strategy, this is always equal to the top_k value.
        
        Args:
            config: RAG configuration
            
        Returns:
            Expected number of results (always top_k)
            
        Example:
            >>> strategy = TopKStrategy()
            >>> config = RAGConfig(top_k=25)
            >>> print(strategy.get_expected_result_count(config))
            25
        """
        return config.top_k
