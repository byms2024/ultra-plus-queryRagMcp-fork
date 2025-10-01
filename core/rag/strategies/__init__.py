"""
RAG Strategy Module.

This module implements the Strategy Pattern for RAG retrieval operations.
It provides a flexible architecture for different retrieval approaches.

Strategy Pattern Benefits:
- Easy to add new retrieval strategies
- Clean separation of concerns
- Testable and maintainable code
- Runtime strategy selection

Available Strategies:
- TopKStrategy: Traditional top-k similarity search
- HybridStrategy: Quality-first with intelligent fallback
"""

from .base_strategy import BaseRetrievalStrategy
from .strategy_factory import StrategyFactory

__all__ = ['BaseRetrievalStrategy', 'StrategyFactory']
