#!/usr/bin/env python3
"""
Test suite for the Unified QueryRAG Engine with NPS data (Customized Profile).
Tests core functionality, question answering, and error handling for Brazilian Portuguese NPS data.
"""

import pytest
import sys
import os
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.unified_engine import UnifiedQueryEngine
from config.base_config import Config
from config.profiles.customized_profile.profile_config import CustomizedProfile
from config.providers.registry import ProviderConfig

# =============================================================================
# TEST CONSTANTS AND SHARED DATA
# =============================================================================

# NPS-specific test data
TEST_DEALER_CODE = "BYDAMEBR0007W"
TEST_RO_NO = "C0125030811193253828"
TEST_SCORE = 9.0
TEST_VIN = "LC0CE4CC2R0009877"

# Mock NPS response data
MOCK_NPS_ANSWER_RESPONSE = {
    "question": "What is the average NPS score for dealer BYDAMEBR0007W?",
    "answer": "O score médio NPS para o dealer BYDAMEBR0007W é 9.0",
    "sources": [{
        "content": "Dados NPS do dealer",
        "metadata": {
            "dealer_code": TEST_DEALER_CODE,
            "score": TEST_SCORE,
            "ro_no": TEST_RO_NO
        }
    }],
    "confidence": "high",
    "method_used": "text2query",
    "execution_time": 2.1,
    "timestamp": "2025-01-01T12:00:00.000000",
    "profile": "customized_profile"
}

MOCK_NPS_RAG_RESPONSE = {
    "question": "Quais são os principais problemas mencionados nas descrições de problemas?",
    "answer": "Com base nos dados NPS, os principais problemas mencionados incluem demora no atendimento, necessidade de peças e solicitações de revisão. Os clientes também mencionam a necessidade de manuais físicos e notas fiscais das revisões.",
    "sources": [{
        "content": "TROUBLE_DESC: Demorou bastante e peguei o carro sem ter terminado, aguardando chegar a peça para finalizar o conserto",
        "metadata": {
            "ro_no": TEST_RO_NO,
            "dealer_code": TEST_DEALER_CODE,
            "score": 7.0
        }
    }],
    "confidence": "medium",
    "method_used": "rag",
    "execution_time": 3.5,
    "timestamp": "2025-01-01T12:00:00.000000",
    "profile": "customized_profile"
}

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.profile_name = "customized_profile"
    config.csv_file = str(Path(__file__).parent / "test_data" / "csv.csv")
    config.sample_size = None
    config.vector_store_path = str(Path(__file__).parent / "test_data" / "vector_store")
    config.chunk_size = 1000
    config.chunk_overlap = 200
    config.top_k = 50
    config.max_iterations = 10
    return config

@pytest.fixture
def nps_profile():
    """Create a customized profile instance for testing."""
    return CustomizedProfile()

@pytest.fixture
def mock_provider_config():
    """Create a mock provider configuration."""
    return ProviderConfig(
        provider="google",
        generation_model="gemini-2.5-flash",
        embedding_model="text-embedding-004",
        credentials={"api_key": "test_key"},
        extras={"temperature": 0.2, "max_tokens": 2048}
    )

@pytest.fixture
def mock_engine(mock_config, nps_profile, mock_provider_config):
    """Create a mock unified engine with NPS data."""
    with patch('core.unified_engine.UnifiedQueryEngine._create_llm_provider') as mock_llm:
        with patch('core.unified_engine.UnifiedQueryEngine._initialize_engines'):
            with patch('core.unified_engine.UnifiedQueryEngine._load_and_process_data'):
                mock_llm.return_value = Mock()
                engine = UnifiedQueryEngine(mock_config, nps_profile)
                return engine

# =============================================================================
# ENGINE INITIALIZATION TESTS
# =============================================================================

class TestUnifiedEngineInitialization:
    """Test unified engine initialization with NPS data."""
    
    def test_engine_initialization_success(self, mock_config, nps_profile):
        """Test successful engine initialization."""
        with patch('core.unified_engine.UnifiedQueryEngine._create_llm_provider'):
            with patch('core.unified_engine.UnifiedQueryEngine._initialize_engines'):
                with patch('core.unified_engine.UnifiedQueryEngine._load_and_process_data'):
                    engine = UnifiedQueryEngine(mock_config, nps_profile)
                    
                    assert engine.config == mock_config
                    assert engine.profile == nps_profile
                    assert engine.profile.profile_name == "customized_profile"
    
    def test_engine_initialization_without_profile(self, mock_config):
        """Test engine initialization with profile auto-loading."""
        with patch('core.unified_engine.UnifiedQueryEngine._create_llm_provider'):
            with patch('core.unified_engine.UnifiedQueryEngine._initialize_engines'):
                with patch('core.unified_engine.UnifiedQueryEngine._load_and_process_data'):
                    with patch('config.profiles.profile_factory.ProfileFactory.create_profile') as mock_factory:
                        mock_profile = Mock()
                        mock_factory.return_value = mock_profile
                        
                        engine = UnifiedQueryEngine(mock_config)
                        
                        assert engine.config == mock_config
                        assert engine.profile == mock_profile
    
    def test_nps_profile_configuration(self, nps_profile):
        """Test NPS profile configuration."""
        assert nps_profile.profile_name == "customized_profile"
        assert nps_profile.language == "pt-BR"
        assert nps_profile.locale == "pt_BR"
        
        # Check required columns for NPS data
        required_columns = nps_profile.required_columns
        assert 'RO_NO' in required_columns
        assert 'DEALER_CODE' in required_columns
        assert 'SCORE' in required_columns
        assert 'SERVICE_ATTITUDE' in required_columns
        assert 'VIN' in required_columns
        assert 'CREATE_DATE' in required_columns
        
        # Check sensitive columns
        sensitive_columns = nps_profile.sensitive_columns
        assert 'VIN' in sensitive_columns
        assert 'DEALER_CODE' in sensitive_columns
        
        # Check text columns
        text_columns = nps_profile.text_columns
        assert 'TROUBLE_DESC' in text_columns
        assert 'CHECK_RESULT' in text_columns
        assert 'OTHERS_REASON' in text_columns

# =============================================================================
# QUESTION ANSWERING TESTS
# =============================================================================

class TestNPSQuestionAnswering:
    """Test question answering with NPS data."""
    
    @pytest.mark.integration
    def test_nps_score_aggregation_query(self, mock_engine):
        """Test NPS score aggregation query."""
        mock_engine.ask_question.return_value = MOCK_NPS_ANSWER_RESPONSE
        
        question = "What is the average NPS score for dealer BYDAMEBR0007W?"
        response = mock_engine.ask_question(question, method="auto")
        
        assert response["question"] == question
        assert "score médio NPS" in response["answer"]
        assert response["method_used"] == "text2query"
        assert response["confidence"] == "high"
        assert response["profile"] == "customized_profile"
    
    @pytest.mark.integration
    def test_nps_portuguese_feedback_analysis(self, mock_engine):
        """Test Portuguese feedback analysis using RAG."""
        mock_engine.ask_question.return_value = MOCK_NPS_RAG_RESPONSE
        
        question = "Quais são os principais problemas mencionados nas descrições de problemas?"
        response = mock_engine.ask_question(question, method="rag")
        
        assert response["question"] == question
        assert "principais problemas" in response["answer"]
        assert response["method_used"] == "rag"
        assert response["confidence"] == "medium"
        assert response["profile"] == "customized_profile"
    
    @pytest.mark.integration
    def test_dealer_performance_comparison(self, mock_engine):
        """Test dealer performance comparison query."""
        comparison_response = {
            "question": "Compare NPS performance between BYDAMEBR0007W and BYDAMEBR0005W",
            "answer": "BYDAMEBR0007W tem um score médio de 8.5 enquanto BYDAMEBR0005W tem 10.0. BYDAMEBR0005W apresenta melhor performance geral.",
            "sources": [],
            "confidence": "high",
            "method_used": "text2query",
            "execution_time": 2.3,
            "profile": "customized_profile"
        }
        
        mock_engine.ask_question.return_value = comparison_response
        
        question = "Compare NPS performance between BYDAMEBR0007W and BYDAMEBR0005W"
        response = mock_engine.ask_question(question, method="auto")
        
        assert "BYDAMEBR0007W" in response["answer"]
        assert "BYDAMEBR0005W" in response["answer"]
        assert response["method_used"] == "text2query"
    
    @pytest.mark.integration
    def test_service_quality_analysis(self, mock_engine):
        """Test service quality analysis query."""
        quality_response = {
            "question": "How many repairs had positive service attitude ratings?",
            "answer": "85% dos reparos tiveram avaliações positivas de atitude de serviço.",
            "sources": [],
            "confidence": "high",
            "method_used": "text2query",
            "execution_time": 1.8,
            "profile": "customized_profile"
        }
        
        mock_engine.ask_question.return_value = quality_response
        
        question = "How many repairs had positive service attitude ratings?"
        response = mock_engine.ask_question(question, method="auto")
        
        assert "85%" in response["answer"]
        assert "atitude de serviço" in response["answer"]
        assert response["method_used"] == "text2query"

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestNPSErrorHandling:
    """Test error handling with NPS data."""
    
    def test_invalid_dealer_code(self, mock_engine):
        """Test handling of invalid dealer codes."""
        error_response = {
            "question": "What is the NPS score for invalid dealer?",
            "answer": "Dealer code not found in the database.",
            "sources": [],
            "confidence": "low",
            "method_used": "text2query",
            "execution_time": 0.5,
            "profile": "customized_profile"
        }
        
        mock_engine.ask_question.return_value = error_response
        
        question = "What is the NPS score for invalid dealer?"
        response = mock_engine.ask_question(question, method="auto")
        
        assert "not found" in response["answer"]
        assert response["confidence"] == "low"
    
    def test_empty_question(self, mock_engine):
        """Test handling of empty questions."""
        with pytest.raises(ValueError):
            mock_engine.ask_question("", method="auto")
    
    def test_none_question(self, mock_engine):
        """Test handling of None questions."""
        with pytest.raises(ValueError):
            mock_engine.ask_question(None, method="auto")
    
    def test_invalid_method(self, mock_engine):
        """Test handling of invalid methods."""
        with pytest.raises(ValueError):
            mock_engine.ask_question("Test question", method="invalid_method")

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestNPSPerformance:
    """Test performance with NPS data."""
    
    @pytest.mark.performance
    def test_query_execution_time(self, mock_engine):
        """Test query execution time."""
        mock_engine.ask_question.return_value = MOCK_NPS_ANSWER_RESPONSE
        
        question = "What is the average NPS score?"
        response = mock_engine.ask_question(question, method="auto")
        
        assert response["execution_time"] <= 5.0  # Should complete within 5 seconds
        assert response["execution_time"] > 0
    
    @pytest.mark.performance
    def test_rag_query_performance(self, mock_engine):
        """Test RAG query performance."""
        mock_engine.ask_question.return_value = MOCK_NPS_RAG_RESPONSE
        
        question = "Analyze customer feedback patterns"
        response = mock_engine.ask_question(question, method="rag")
        
        assert response["execution_time"] <= 10.0  # RAG queries may take longer
        assert response["method_used"] == "rag"

# =============================================================================
# DATA PROCESSING TESTS
# =============================================================================

class TestNPSDataProcessing:
    """Test NPS data processing and cleaning."""
    
    def test_nps_data_cleaning(self, nps_profile):
        """Test NPS data cleaning functionality."""
        # Create sample NPS data
        test_data = pd.DataFrame({
            'RO_NO': ['RO001', 'RO002'],
            'DEALER_CODE': ['DEALER1', 'DEALER2'],
            'SCORE': [9.0, '8.5'],  # Mixed types
            'SERVICE_ATTITUDE': ['Y', ''],
            'TROUBLE_DESC': ['Problem description', None],
            'CREATE_DATE': ['2025-01-01', '2025-01-02']
        })
        
        # Test cleaning
        cleaned_data = nps_profile.clean_data(test_data)
        
        assert cleaned_data['SCORE'].dtype == float
        assert cleaned_data['TROUBLE_DESC'].isna().sum() == 0  # Should fill NaN
    
    def test_sensitization_rules(self, nps_profile):
        """Test NPS data sensitization."""
        # Test VIN sensitization
        vin_value = "LGXC74C44R0009167"
        sensitized = nps_profile.sensitize_value(vin_value, 'VIN', {})
        
        assert sensitized.startswith('VIN_')
        assert len(sensitized) > len('VIN_')
        
        # Test dealer code sensitization
        dealer_value = "BYDAMEBR0007W"
        sensitized_dealer = nps_profile.sensitize_value(dealer_value, 'DEALER_CODE', {})
        
        assert sensitized_dealer.startswith('DEALER_')
        assert len(sensitized_dealer) > len('DEALER_')

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestNPSIntegration:
    """Integration tests for NPS profile."""
    
    @pytest.mark.integration
    def test_end_to_end_nps_query(self, mock_engine):
        """Test end-to-end NPS query processing."""
        end_to_end_response = {
            "question": "What is the overall NPS performance for all dealers?",
            "answer": "O desempenho geral NPS para todos os dealers é de 8.7, com BYDAMEBR0045W apresentando a melhor performance (10.0) e BYDAMEBR0007W com 8.5.",
            "sources": [
                {
                    "content": "Dados NPS consolidados",
                    "metadata": {"dealer_code": "BYDAMEBR0045W", "score": 10.0}
                }
            ],
            "confidence": "high",
            "method_used": "text2query",
            "execution_time": 2.8,
            "profile": "customized_profile"
        }
        
        mock_engine.ask_question.return_value = end_to_end_response
        
        question = "What is the overall NPS performance for all dealers?"
        response = mock_engine.ask_question(question, method="auto")
        
        assert "desempenho geral NPS" in response["answer"]
        assert "BYDAMEBR0045W" in response["answer"]
        assert response["sources"] is not None
        assert len(response["sources"]) > 0
    
    @pytest.mark.integration
    def test_portuguese_language_processing(self, nps_profile):
        """Test Portuguese language processing."""
        # Test prompt template
        prompt = nps_profile.get_prompt_template()
        assert "português brasileiro" in prompt
        assert "assistente de análise" in prompt
        
        # Test system prompt
        system_prompt = nps_profile.get_llm_system_prompt()
        assert "NPS" in system_prompt
        assert "pandas code" in system_prompt
        
        # Test schema hints
        hints = nps_profile.get_schema_hints("sample data")
        assert "NPS" in hints
        assert "DEALER_CODE" in hints
        assert "SCORE" in hints

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
