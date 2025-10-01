"""
Hybrid Retrieval Strategy Implementation.

This module implements the hybrid retrieval strategy that combines quality-first
filtering with intelligent fallback mechanisms. It provides the best balance
between result quality and reliability.

Strategy Characteristics:
- Quality-first: Prioritizes high-similarity results
- Adaptive: Adjusts to query relevance
- Reliable: Never returns empty results
- Balanced: Combines quality and quantity
- Best For: Queries where quality is important
"""

from typing import List, Dict, Any, TYPE_CHECKING

from .base_strategy import BaseRetrievalStrategy

if TYPE_CHECKING:
    from core.rag.generic_vector_store import GenericVectorStore
    from core.rag.config.rag_config import RAGConfig


class HybridStrategy(BaseRetrievalStrategy):
    """
    Hybrid retrieval strategy: Quality-first with intelligent fallback.
    
    This strategy implements a sophisticated retrieval algorithm that balances
    quality and quantity through a multi-step process:
    
    Algorithm:
    1. Search for many candidates (max_search_with_threshold)
    2. Filter by quality threshold (similarity_threshold)
    3. If insufficient results, fallback to best available (B4.2 approach)
    
    The fallback mechanism (B4.2) sorts existing candidates by similarity
    and returns the top min_results_with_threshold, which is faster than
    performing a new search (B4.1 approach).
    
    Performance Characteristics:
    - Latency: Medium (~100ms for typical datasets)
    - Memory Usage: Medium (stores candidates + filtered results)
    - Quality: High (prioritizes high-similarity results)
    - Predictability: Medium (varies with query relevance)
    - Scalability: Good (controlled by max_search_with_threshold)
    
    Example:
        >>> strategy = HybridStrategy()
        >>> config = RAGConfig(
        ...     retrieval_strategy="hybrid",
        ...     similarity_threshold=0.7,
        ...     max_search_with_threshold=100,
        ...     min_results_with_threshold=3
        ... )
        >>> results = strategy.search_relevant_chunks(
        ...     query="Samsung refrigerators",
        ...     vectorstore=vectorstore,
        ...     config=config
        ... )
        >>> print(f"Found {len(results)} high-quality results")
    """
    
    def search_relevant_chunks(
        self, 
        query: str, 
        vectorstore: 'GenericVectorStore', 
        config: 'RAGConfig'
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using hybrid strategy.
        
        This method implements the hybrid retrieval algorithm:
        1. B2: Search for many candidates first
        2. B3: Filter by quality threshold
        3. B4: Fallback if insufficient high-quality results
        
        Args:
            query: User query string to search for
            vectorstore: Vector store instance containing document embeddings
            config: RAG configuration with hybrid parameters
            
        Returns:
            List of result dictionaries, each containing:
            - content: Document content (str)
            - similarity: Similarity score (float, 0.0-1.0)
            - metadata: Document metadata (dict, optional)
            
        Example:
            >>> strategy = HybridStrategy()
            >>> config = RAGConfig(
            ...     similarity_threshold=0.8,
            ...     max_search_with_threshold=150,
            ...     min_results_with_threshold=5
            ... )
            >>> results = strategy.search_relevant_chunks(
            ...     "Samsung refrigerators", vectorstore, config
            ... )
            >>> # Results will have similarity >= 0.8 or be top 5 available
        """
        # B2. Search for many candidates first
        # This ensures we have a broad pool to filter from
        # Using max_search_with_threshold to get sufficient candidates
        candidates = vectorstore.similarity_search_with_score(
            query, k=config.max_search_with_threshold
        )
        
        # B3. Filter by quality threshold
        # Only keep results that meet our quality standards
        # This is the core quality-first filtering step
        filtered_results = [
            (doc, score) for doc, score in candidates 
            if score >= config.similarity_threshold
        ]
        
        # B4. Fallback if insufficient high-quality results
        # This ensures we never return empty results
        if len(filtered_results) < config.min_results_with_threshold:
            # B4.2: Sort existing candidates by similarity (faster than new search)
            # This approach is preferred over B4.1 because:
            # - No additional vector search needed
            # - Reuses already computed similarity scores
            # - Faster execution time
            sorted_candidates = sorted(
                candidates, 
                key=lambda x: x[1],  # Sort by similarity score (index 1)
                reverse=True  # Highest similarity first
            )
            filtered_results = sorted_candidates[:config.min_results_with_threshold]
        
        # Convert to standard result format
        # Each result is a tuple of (Document, similarity_score)
        formatted_results = []
        for doc, score in filtered_results:
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
            String name "hybrid"
            
        Example:
            >>> strategy = HybridStrategy()
            >>> print(strategy.get_strategy_name())
            "hybrid"
        """
        return "hybrid"
    
    def get_strategy_description(self) -> str:
        """
        Get a human-readable description of this strategy.
        
        Returns:
            String description explaining the hybrid approach
            
        Example:
            >>> strategy = HybridStrategy()
            >>> print(strategy.get_strategy_description())
            "Hybrid strategy: Quality-first with intelligent fallback mechanisms"
        """
        return "Hybrid strategy: Quality-first with intelligent fallback mechanisms"
    
    def get_performance_characteristics(self) -> Dict[str, Any]:
        """
        Get performance characteristics specific to hybrid strategy.
        
        Returns:
            Dictionary containing hybrid-specific performance metrics
            
        Example:
            >>> strategy = HybridStrategy()
            >>> perf = strategy.get_performance_characteristics()
            >>> print(f"Quality: {perf['quality']}")
            >>> print(f"Latency: {perf['latency']}")
        """
        return {
            "latency": "medium",        # Two-step process: search + filter
            "memory_usage": "medium",   # Stores candidates + filtered results
            "quality": "high",          # Prioritizes high-similarity results
            "predictability": "medium", # Varies with query relevance
            "scalability": "good"       # Controlled by max_search_with_threshold
        }
    
    def validate_config(self, config: 'RAGConfig') -> None:
        """
        Validate that the configuration is compatible with hybrid strategy.
        
        For hybrid strategy, we need to validate the relationship between
        similarity_threshold, max_search_with_threshold, and min_results_with_threshold.
        
        Args:
            config: RAG configuration to validate
            
        Raises:
            ValueError: If configuration is invalid for hybrid strategy
            
        Example:
            >>> strategy = HybridStrategy()
            >>> config = RAGConfig(
            ...     retrieval_strategy="hybrid",
            ...     similarity_threshold=0.7,
            ...     max_search_with_threshold=100,
            ...     min_results_with_threshold=5
            ... )
            >>> strategy.validate_config(config)  # Should pass
            
            >>> bad_config = RAGConfig(min_results_with_threshold=200, max_search_with_threshold=100)
            >>> strategy.validate_config(bad_config)  # Should raise ValueError
        """
        # Call base validation first
        super().validate_config(config)
        
        # Additional hybrid-specific validation
        if config.min_results_with_threshold > config.max_search_with_threshold:
            raise ValueError(
                f"min_results_with_threshold ({config.min_results_with_threshold}) "
                f"cannot be greater than max_search_with_threshold ({config.max_search_with_threshold})"
            )
        
        # Validate reasonable thresholds for hybrid strategy
        if config.similarity_threshold < 0.1:
            raise ValueError(
                f"similarity_threshold should be at least 0.1 for hybrid strategy, "
                f"got {config.similarity_threshold}"
            )
        
        if config.similarity_threshold > 0.95:
            raise ValueError(
                f"similarity_threshold should not exceed 0.95 for hybrid strategy, "
                f"got {config.similarity_threshold}"
            )
    
    def get_expected_result_count_range(self, config: 'RAGConfig') -> tuple[int, int]:
        """
        Get the expected range of result counts for this strategy.
        
        For hybrid strategy, the result count varies based on query relevance:
        - Minimum: min_results_with_threshold (fallback case)
        - Maximum: Number of results above similarity_threshold (up to max_search_with_threshold)
        
        Args:
            config: RAG configuration
            
        Returns:
            Tuple of (min_expected, max_expected) result counts
            
        Example:
            >>> strategy = HybridStrategy()
            >>> config = RAGConfig(
            ...     similarity_threshold=0.7,
            ...     max_search_with_threshold=100,
            ...     min_results_with_threshold=5
            ... )
            >>> min_count, max_count = strategy.get_expected_result_count_range(config)
            >>> print(f"Expected {min_count}-{max_count} results")
        """
        min_expected = config.min_results_with_threshold
        max_expected = config.max_search_with_threshold
        return (min_expected, max_expected)
    
    def analyze_query_relevance(self, query: str, results: List[Dict[str, Any]], config: 'RAGConfig') -> Dict[str, Any]:
        """
        Analyze the relevance of query results for this strategy.
        
        This method provides insights into how well the hybrid strategy
        performed for a given query.
        
        Args:
            query: Original query string
            results: Retrieved results from search_relevant_chunks
            config: RAG configuration used
            
        Returns:
            Dictionary containing relevance analysis:
            - total_results: Total number of results
            - high_quality_results: Results above similarity_threshold
            - fallback_used: Whether fallback mechanism was used
            - avg_similarity: Average similarity score
            - quality_ratio: Ratio of high-quality results
            
        Example:
            >>> strategy = HybridStrategy()
            >>> config = RAGConfig(similarity_threshold=0.7)
            >>> results = strategy.search_relevant_chunks("query", vectorstore, config)
            >>> analysis = strategy.analyze_query_relevance("query", results, config)
            >>> print(f"Quality ratio: {analysis['quality_ratio']:.2f}")
        """
        total_results = len(results)
        high_quality_results = sum(1 for r in results if r['similarity'] >= config.similarity_threshold)
        fallback_used = total_results < config.min_results_with_threshold or high_quality_results < config.min_results_with_threshold
        
        avg_similarity = sum(r['similarity'] for r in results) / total_results if results else 0.0
        quality_ratio = high_quality_results / total_results if total_results > 0 else 0.0
        
        return {
            "total_results": total_results,
            "high_quality_results": high_quality_results,
            "fallback_used": fallback_used,
            "avg_similarity": avg_similarity,
            "quality_ratio": quality_ratio
        }
