#!/usr/bin/env python3
"""
Unified Query Engine that combines Text2Query and RAG approaches.
Implements the requested workflow:
1. Load test data from active profile to pandas DataFrame
2. Try Text2Query functionality first
3. If no result, fallback to RAG logic
4. Return unified response
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import time
from datetime import datetime

from config.base_config import load_system_config, get_profile
from config.logging_config import get_logger
from core.text2query.engine import QuerySynthesisEngine
from core.rag.generic_rag_agent import GenericRAGAgent
from core.rag.generic_data_processor import DataSchema

logger = get_logger(__name__)


class UnifiedQueryEngine:
    """
    Unified engine that orchestrates between Text2Query and RAG approaches.
    
    Workflow:
    1. Load data from active profile
    2. Try Text2Query first (direct pandas querying)
    3. Fallback to RAG if Text2Query yields no result
    4. Return unified response format
    """
    
    def __init__(self, profile_name: Optional[str] = None):
        """Initialize the unified engine with the specified profile."""
        self.config = load_system_config()
        self.profile = get_profile()
        
        # Override profile if specified
        if profile_name:
            from config.profiles.profile_factory import ProfileFactory
            self.profile = ProfileFactory.create_profile(profile_name)
        
        # Load and process data
        self.df = self._load_data()
        
        # Initialize engines
        self.text2query_engine = None
        self.rag_agent = None
        
        self._initialize_engines()
        
        logger.info(f"Unified engine initialized with profile: {self.profile.profile_name}")
        logger.info(f"Data loaded: {len(self.df)} records, {len(self.df.columns)} columns")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from the active profile."""
        try:
            # Use the profile's data file path
            data_file = self.profile.get_data_file_path()
            logger.info(f"Loading data from: {data_file}")
            
            # Load CSV data
            df = pd.read_csv(data_file)
            
            # Clean data using profile's cleaning method
            if hasattr(self.profile, 'clean_data'):
                df = self.profile.clean_data(df)
            
            logger.info(f"Data loaded successfully: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _initialize_engines(self):
        """Initialize both Text2Query and RAG engines."""
        try:
            # Initialize Text2Query engine
            from config.base_config import Config
            text2query_config = Config(
                google_api_key="PLACEHOLDER",  # Will be loaded from profile
                generation_model=self.config.generation_model,
                port=self.config.api_port,
                profile_name=self.profile.profile_name
            )
            
            self.text2query_engine = QuerySynthesisEngine(text2query_config, self.profile)
            logger.info("Text2Query engine initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Text2Query engine: {e}")
            self.text2query_engine = None
        
        try:
            # Initialize RAG agent
            data_schema = self.profile.get_data_schema()
            collection_name = f"{self.profile.profile_name}_data"
            provider_config = self.profile.get_provider_config()
            
            # Create LangChain config for RAG
            from config.langchain_settings import LangChainConfig
            langchain_config = LangChainConfig()
            
            self.rag_agent = GenericRAGAgent(
                langchain_config, 
                data_schema, 
                collection_name, 
                provider_config
            )
            logger.info("RAG agent initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize RAG agent: {e}")
            self.rag_agent = None
    
    def answer_question(self, question: str, method: str = "auto") -> Dict[str, Any]:
        """
        Answer a question using the unified approach.
        
        Args:
            question: Natural language question
            method: "auto", "text2query", "rag", or "both"
        
        Returns:
            Dictionary with answer, sources, method used, and metadata
        """
        logger.info(f"Processing question: {question}")
        start_time = time.time()
        
        result = None
        method_used = None
        
        try:
            if method == "auto" or method == "text2query":
                # Try Text2Query first
                if self.text2query_engine:
                    logger.info("Attempting Text2Query approach...")
                    result = self.text2query_engine.execute_query(question)
                    
                    if result and not result.get('error') and result.get('answer'):
                        method_used = "text2query"
                        logger.info("Text2Query succeeded")
                    else:
                        logger.info("Text2Query yielded no result, will try RAG")
                        result = None
                else:
                    logger.warning("Text2Query engine not available")
            
            # Fallback to RAG if Text2Query didn't work
            if not result and (method == "auto" or method == "rag"):
                if self.rag_agent:
                    logger.info("Attempting RAG approach...")
                    rag_result = self.rag_agent.answer_question(question)
                    
                    if rag_result and rag_result.get('answer'):
                        result = self._convert_rag_result(rag_result)
                        method_used = "rag"
                        logger.info("RAG succeeded")
                    else:
                        logger.warning("RAG also yielded no result")
                else:
                    logger.warning("RAG agent not available")
            
            # If still no result, return error
            if not result:
                result = {
                    "answer": "I'm sorry, I couldn't find an answer to your question. Please try rephrasing it or providing more specific details.",
                    "sources": [],
                    "confidence": "low",
                    "error": "No result from either method"
                }
                method_used = "none"
            
            # Add metadata
            execution_time = time.time() - start_time
            result.update({
                "method_used": method_used,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "profile": self.profile.profile_name
            })
            
            logger.info(f"Question answered using {method_used} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": f"Error processing question: {e}",
                "sources": [],
                "confidence": "low",
                "method_used": "error",
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "profile": self.profile.profile_name,
                "error": str(e)
            }
    
    def _convert_rag_result(self, rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert RAG result to unified format."""
        return {
            "answer": rag_result.get("answer", ""),
            "sources": rag_result.get("sources", []),
            "confidence": rag_result.get("confidence", "medium")
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from both engines."""
        stats = {
            "profile": self.profile.profile_name,
            "data": {
                "total_records": len(self.df),
                "total_columns": len(self.df.columns),
                "column_names": list(self.df.columns)
            },
            "engines": {
                "text2query_available": self.text2query_engine is not None,
                "rag_available": self.rag_agent is not None
            }
        }
        
        # Add Text2Query stats if available
        if self.text2query_engine:
            try:
                text2query_stats = self.text2query_engine.get_stats()
                stats["text2query"] = text2query_stats
            except Exception as e:
                logger.warning(f"Failed to get Text2Query stats: {e}")
        
        # Add RAG stats if available
        if self.rag_agent:
            try:
                rag_stats = self.rag_agent.get_stats()
                stats["rag"] = rag_stats
            except Exception as e:
                logger.warning(f"Failed to get RAG stats: {e}")
        
        return stats
    
    def search_data(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant data chunks using RAG."""
        if not self.rag_agent:
            logger.warning("RAG agent not available for search")
            return []
        
        try:
            return self.rag_agent.search_relevant_chunks(query, top_k)
        except Exception as e:
            logger.error(f"Error searching data: {e}")
            return []
    
    def rebuild_rag_index(self) -> bool:
        """Rebuild the RAG vector store."""
        if not self.rag_agent:
            logger.warning("RAG agent not available for rebuild")
            return False
        
        try:
            return self.rag_agent.rebuild_vectorstore()
        except Exception as e:
            logger.error(f"Error rebuilding RAG index: {e}")
            return False
    
    def get_available_methods(self) -> List[str]:
        """Get list of available query methods."""
        methods = []
        if self.text2query_engine:
            methods.append("text2query")
        if self.rag_agent:
            methods.append("rag")
        return methods
