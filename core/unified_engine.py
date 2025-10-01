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
import re
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
    
    def _should_use_rag_directly(self, question: str) -> bool:
        """
        Intelligent question classification to determine if RAG should be used directly.
        
        Skips Text2Query for questions that are clearly semantic/qualitative in nature.
        Text2Query is best for quantitative queries (counts, aggregations, filtering).
        RAG is best for qualitative queries (explanations, descriptions, summaries).
        
        Returns:
            True if the question should go directly to RAG, False if Text2Query should be tried first
        """
        question_lower = question.lower()
        
        # Semantic/qualitative question patterns (English)
        rag_patterns_en = [
            r'\bwhat (are|is) (the )?(main|primary|key|common|typical|most)',  # "What are the main..."
            r'\btell me (about|what)',                                          # "Tell me about..."
            r'\bexplain',                                                        # "Explain..."
            r'\bdescribe',                                                       # "Describe..."
            r'\bsummarize',                                                      # "Summarize..."
            r'\bsummary of',                                                     # "Summary of..."
            r'\blist (the )?(main|primary|key|all)',                            # "List the main..."
            r'\bwhat does .+ mean',                                              # "What does X mean?"
            r'\bwhy (do|does|did|is|are)',                                      # "Why..."
            r'\bhow (do|does|did|can|should) (i|we|they)',                      # "How do/can..."
            r'\bprovide (a |an )?(brief )?(overview|summary)',                  # "Provide overview..."
            r'\bgive me (a |an )?(brief )?(overview|summary|description)',      # "Give me..."
            r'\bshow me .+ (about|related to)',                                 # "Show me info about..."
        ]
        
        # Semantic/qualitative question patterns (Portuguese)
        rag_patterns_pt = [
            r'\bquais s[ãa]o (os|as) (principais|mais|comuns)',                # "Quais são os principais..."
            r'\bme (diga|conte|explique|mostre)',                               # "Me diga/conte..."
            r'\bexplique',                                                       # "Explique..."
            r'\bdescreva',                                                       # "Descreva..."
            r'\bresumo',                                                         # "Resumo..."
            r'\blistar (os|as) (principais|todos)',                             # "Listar os principais..."
            r'\bo que (significa|s[ãa]o)',                                      # "O que significa/são..."
            r'\bpor que',                                                        # "Por que..."
            r'\bcomo (posso|podemos|fazer)',                                    # "Como posso/podemos..."
            r'\bforne[çc]a (um |uma )?(breve )?(vis[ãa]o|resumo)',             # "Forneça visão..."
            r'\bme d[êe] (um |uma )?(breve )?(vis[ãa]o|resumo|descri[çc][ãa]o)',  # "Me dê..."
            r'\bmostre.+(sobre|relacionado)',                                   # "Mostre sobre..."
        ]
        
        # Quantitative question patterns that should use Text2Query
        text2query_patterns = [
            r'\bhow many',                                                       # "How many..."
            r'\bcount',                                                          # "Count..."
            r'\baverage',                                                        # "Average..."
            r'\bmean',                                                           # "Mean..."
            r'\bmedian',                                                         # "Median..."
            r'\bsum of',                                                         # "Sum of..."
            r'\btotal',                                                          # "Total..."
            r'\btop \d+',                                                        # "Top 10..."
            r'\bfirst \d+',                                                      # "First 5..."
            r'\blast \d+',                                                       # "Last 3..."
            r'\bgreater than',                                                   # "Greater than..."
            r'\bless than',                                                      # "Less than..."
            r'\bbetween .+ and',                                                 # "Between X and Y..."
            r'\bquantos',                                                        # "Quantos..." (Portuguese)
            r'\bcontar',                                                         # "Contar..." (Portuguese)
            r'\bm[ée]dia',                                                      # "Média..." (Portuguese)
            r'\bsoma',                                                           # "Soma..." (Portuguese)
            r'\btotal',                                                          # "Total..." (Portuguese)
            r'\bprimeiros \d+',                                                  # "Primeiros 10..." (Portuguese)
            r'\b[úu]ltimos \d+',                                                # "Últimos 5..." (Portuguese)
        ]
        
        # Check if it's a quantitative question (use Text2Query)
        for pattern in text2query_patterns:
            if re.search(pattern, question_lower):
                logger.info(f"📊 Quantitative pattern detected: using Text2Query first")
                return False
        
        # Check if it's a semantic question (use RAG)
        for pattern in rag_patterns_en + rag_patterns_pt:
            if re.search(pattern, question_lower):
                logger.info(f"🎯 Semantic pattern detected: routing directly to RAG")
                return True
        
        # Default: try Text2Query first (conservative approach)
        return False
    
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
            # Intelligent routing: skip Text2Query for semantic questions
            skip_text2query = False
            if method == "auto":
                skip_text2query = self._should_use_rag_directly(question)
            
            if (method == "auto" or method == "text2query") and not skip_text2query:
                # Try Text2Query first
                if self.text2query_engine:
                    text2query_start = time.time()
                    budget_seconds = getattr(self.config, "text2query_time_budget_seconds", 120)
                    logger.info(f"⏳ Attempting Text2Query approach (budget: {budget_seconds}s)...")
                    
                    # Simple time-aware execution with periodic checks
                    result = self.text2query_engine.execute_query(question)
                    text2query_duration = time.time() - text2query_start
                    
                    # Check if we exceeded budget after execution
                    if text2query_duration > budget_seconds:
                        logger.warning(f"⚠️ Text2Query exceeded budget ({text2query_duration:.2f}s > {budget_seconds}s)")
                    
                    if result and not result.get('error') and result.get('answer'):
                        # Check if the answer starts with "Error:" - this indicates Text2Query failed
                        answer_text = result.get('answer', '')
                        if answer_text.startswith('Error:'):
                            logger.info(f"❌ Text2Query returned error response after {text2query_duration:.2f}s, will try RAG fallback")
                            logger.info(f"   Error details: {answer_text[:200]}")
                            result = None
                        else:
                            method_used = "text2query"
                            logger.info(f"✅ Text2Query succeeded in {text2query_duration:.2f}s")
                    else:
                        logger.info(f"❌ Text2Query yielded no result after {text2query_duration:.2f}s, will try RAG fallback")
                        result = None
                else:
                    logger.warning("Text2Query engine not available")
            elif skip_text2query:
                logger.info("⚡ Skipping Text2Query for semantic question, going directly to RAG")
            
            # Fallback to RAG if Text2Query didn't work
            if not result and (method == "auto" or method == "rag"):
                if self.rag_agent:
                    rag_start = time.time()
                    logger.info("⏳ Attempting RAG approach (fallback)...")
                    rag_result = self.rag_agent.answer_question(question)
                    rag_duration = time.time() - rag_start
                    
                    if rag_result and rag_result.get('answer'):
                        result = self._convert_rag_result(rag_result)
                        method_used = "rag"
                        logger.info(f"✅ RAG succeeded in {rag_duration:.2f}s")
                    else:
                        logger.warning(f"❌ RAG also yielded no result after {rag_duration:.2f}s")
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
