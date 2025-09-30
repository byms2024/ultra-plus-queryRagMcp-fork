#!/usr/bin/env python3
"""
Test suite for the Unified QueryRAG MCP Server with NPS data (Customized Profile).
Tests MCP tools, integration, and error handling for Brazilian Portuguese NPS data.
"""

import pytest
import sys
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from servers.unified_mcp_server import UnifiedMCPServer
from config.profiles.customized_profile.profile_config import CustomizedProfile

# =============================================================================
# TEST CONSTANTS AND SHARED DATA
# =============================================================================

# NPS-specific test data
TEST_DEALER_CODE = "BYDAMEBR0007W"
TEST_RO_NO = "C0125030811193253828"
TEST_SCORE = 9.0
TEST_VIN = "LC0CE4CC2R0009877"

# Mock NPS MCP responses
MOCK_NPS_MCP_ASK_RESPONSE = {
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
    "profile": "customized_profile"
}

MOCK_NPS_MCP_SEARCH_RESPONSE = {
    "query": "BYDAMEBR0007W",
    "results": [
        {
            "content": "TROUBLE_DESC: Demorou bastante e peguei o carro sem ter terminado",
            "metadata": {
                "ro_no": TEST_RO_NO,
                "dealer_code": TEST_DEALER_CODE,
                "score": TEST_SCORE,
                "vin": TEST_VIN
            },
            "score": 0.85
        }
    ],
    "total_found": 1
}

MOCK_NPS_MCP_STATS_RESPONSE = {
    "profile": "customized_profile",
    "data_records": 3000,
    "engines": {
        "text2query": {
            "available": True,
            "methods": ["traditional", "langchain_direct", "langchain_agent"]
        },
        "rag": {
            "available": True,
            "vector_store": "chroma",
            "documents": 3001
        }
    }
}

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP server instance."""
    with patch('servers.unified_mcp_server.UnifiedMCPServer._initialize_engine'):
        server = UnifiedMCPServer()
        return server

@pytest.fixture
def nps_profile():
    """Create a customized profile instance for testing."""
    return CustomizedProfile()

# =============================================================================
# MCP TOOL TESTS
# =============================================================================

class TestNPSMCPTools:
    """Test MCP tools with NPS data."""
    
    def test_ask_question_tool(self, mock_mcp_server):
        """Test ask_question MCP tool."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.ask_question.return_value = MOCK_NPS_MCP_ASK_RESPONSE
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.ask_question(
                question="What is the average NPS score for dealer BYDAMEBR0007W?",
                method="auto"
            )
            
            assert result["question"] == "What is the average NPS score for dealer BYDAMEBR0007W?"
            assert "score médio NPS" in result["answer"]
            assert result["profile"] == "customized_profile"
            assert result["method_used"] == "text2query"
    
    def test_search_data_tool(self, mock_mcp_server):
        """Test search_data MCP tool."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.search_data.return_value = MOCK_NPS_MCP_SEARCH_RESPONSE
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.search_data(
                query="BYDAMEBR0007W",
                top_k=10
            )
            
            assert result["query"] == "BYDAMEBR0007W"
            assert len(result["results"]) > 0
            assert result["results"][0]["metadata"]["dealer_code"] == TEST_DEALER_CODE
    
    def test_get_stats_tool(self, mock_mcp_server):
        """Test get_stats MCP tool."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.get_stats.return_value = MOCK_NPS_MCP_STATS_RESPONSE
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.get_stats()
            
            assert result["profile"] == "customized_profile"
            assert result["data_records"] == 3000
            assert "text2query" in result["engines"]
            assert "rag" in result["engines"]
    
    def test_get_profile_info_tool(self, mock_mcp_server):
        """Test get_profile_info MCP tool."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.profile.profile_name = "customized_profile"
            mock_engine.profile.language = "pt-BR"
            mock_engine.profile.locale = "pt_BR"
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.get_profile_info()
            
            assert result["profile_name"] == "customized_profile"
            assert result["language"] == "pt-BR"
            assert result["locale"] == "pt_BR"
    
    def test_get_available_methods_tool(self, mock_mcp_server):
        """Test get_available_methods MCP tool."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.get_available_methods.return_value = ["text2query", "rag"]
            mock_engine.profile.profile_name = "customized_profile"
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.get_available_methods()
            
            assert "text2query" in result["available_methods"]
            assert "rag" in result["available_methods"]
            assert result["current_profile"] == "customized_profile"
    
    def test_rebuild_rag_index_tool(self, mock_mcp_server):
        """Test rebuild_rag_index MCP tool."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.rebuild_vector_index.return_value = {"status": "success", "documents_rebuilt": 3001}
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.rebuild_rag_index()
            
            assert result["status"] == "success"
            assert result["documents_rebuilt"] == 3001

# =============================================================================
# NPS-SPECIFIC TOOL TESTS
# =============================================================================

class TestNPSMCPTools:
    """Test NPS-specific MCP tool functionality."""
    
    def test_portuguese_question_handling(self, mock_mcp_server):
        """Test Portuguese question handling."""
        portuguese_response = {
            "question": "Qual é o score médio NPS?",
            "answer": "O score médio NPS é 9.1",
            "sources": [],
            "confidence": "high",
            "method_used": "text2query",
            "execution_time": 2.0,
            "profile": "customized_profile"
        }
        
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.ask_question.return_value = portuguese_response
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.ask_question(
                question="Qual é o score médio NPS?",
                method="auto"
            )
            
            assert "score médio NPS" in result["answer"]
            assert result["profile"] == "customized_profile"
    
    def test_dealer_performance_query(self, mock_mcp_server):
        """Test dealer performance query."""
        dealer_response = {
            "question": "Compare NPS performance between BYDAMEBR0007W and BYDAMEBR0005W",
            "answer": "BYDAMEBR0005W tem melhor performance (10.0) comparado a BYDAMEBR0007W (9.0)",
            "sources": [],
            "confidence": "high",
            "method_used": "text2query",
            "execution_time": 2.3,
            "profile": "customized_profile"
        }
        
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.ask_question.return_value = dealer_response
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.ask_question(
                question="Compare NPS performance between BYDAMEBR0007W and BYDAMEBR0005W",
                method="auto"
            )
            
            assert "BYDAMEBR0005W" in result["answer"]
            assert "BYDAMEBR0007W" in result["answer"]
    
    def test_service_quality_analysis(self, mock_mcp_server):
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
        
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.ask_question.return_value = quality_response
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.ask_question(
                question="How many repairs had positive service attitude ratings?",
                method="auto"
            )
            
            assert "85%" in result["answer"]
            assert "atitude de serviço" in result["answer"]
    
    def test_trouble_description_search(self, mock_mcp_server):
        """Test trouble description search."""
        trouble_search_response = {
            "query": "demora",
            "results": [
                {
                    "content": "TROUBLE_DESC: Demorou bastante e peguei o carro sem ter terminado",
                    "metadata": {
                        "ro_no": TEST_RO_NO,
                        "dealer_code": TEST_DEALER_CODE,
                        "score": 7.0
                    },
                    "score": 0.90
                }
            ],
            "total_found": 1
        }
        
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.search_data.return_value = trouble_search_response
            mock_get_engine.return_value = mock_engine
            
            result = mock_mcp_server.search_data(
                query="demora",
                top_k=10
            )
            
            assert "Demorou" in result["results"][0]["content"]
            assert result["total_found"] == 1

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestNPSMCPErrorHandling:
    """Test error handling in MCP tools with NPS data."""
    
    def test_invalid_question_parameter(self, mock_mcp_server):
        """Test invalid question parameter."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.ask_question.side_effect = ValueError("Invalid question format")
            mock_get_engine.return_value = mock_engine
            
            with pytest.raises(ValueError):
                mock_mcp_server.ask_question(
                    question="",  # Empty question
                    method="auto"
                )
    
    def test_invalid_method_parameter(self, mock_mcp_server):
        """Test invalid method parameter."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.ask_question.side_effect = ValueError("Invalid method")
            mock_get_engine.return_value = mock_engine
            
            with pytest.raises(ValueError):
                mock_mcp_server.ask_question(
                    question="Test question",
                    method="invalid_method"
                )
    
    def test_invalid_search_query(self, mock_mcp_server):
        """Test invalid search query."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.search_data.side_effect = ValueError("Invalid query")
            mock_get_engine.return_value = mock_engine
            
            with pytest.raises(ValueError):
                mock_mcp_server.search_data(
                    query="",  # Empty query
                    top_k=10
                )
    
    def test_engine_initialization_error(self, mock_mcp_server):
        """Test engine initialization error."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_get_engine.side_effect = Exception("Engine initialization failed")
            
            with pytest.raises(Exception):
                mock_mcp_server.ask_question(
                    question="Test question",
                    method="auto"
                )

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestNPSMCPIntegration:
    """Integration tests for NPS MCP server."""
    
    @pytest.mark.integration
    def test_full_nps_mcp_workflow(self, mock_mcp_server):
        """Test full NPS MCP workflow."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.ask_question.return_value = MOCK_NPS_MCP_ASK_RESPONSE
            mock_engine.search_data.return_value = MOCK_NPS_MCP_SEARCH_RESPONSE
            mock_engine.get_stats.return_value = MOCK_NPS_MCP_STATS_RESPONSE
            mock_engine.get_available_methods.return_value = ["text2query", "rag"]
            mock_engine.profile.profile_name = "customized_profile"
            mock_engine.profile.language = "pt-BR"
            mock_engine.rebuild_vector_index.return_value = {"status": "success", "documents_rebuilt": 3001}
            mock_get_engine.return_value = mock_engine
            
            # 1. Ask a question
            ask_result = mock_mcp_server.ask_question(
                question="What is the average NPS score?",
                method="auto"
            )
            assert "score médio NPS" in ask_result["answer"]
            
            # 2. Search for data
            search_result = mock_mcp_server.search_data(
                query="BYDAMEBR0007W",
                top_k=5
            )
            assert len(search_result["results"]) > 0
            
            # 3. Get stats
            stats_result = mock_mcp_server.get_stats()
            assert stats_result["profile"] == "customized_profile"
            
            # 4. Get profile info
            profile_result = mock_mcp_server.get_profile_info()
            assert profile_result["profile_name"] == "customized_profile"
            
            # 5. Get available methods
            methods_result = mock_mcp_server.get_available_methods()
            assert "text2query" in methods_result["available_methods"]
            
            # 6. Rebuild index
            rebuild_result = mock_mcp_server.rebuild_rag_index()
            assert rebuild_result["status"] == "success"
    
    @pytest.mark.integration
    def test_portuguese_language_integration(self, mock_mcp_server):
        """Test Portuguese language integration."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            
            # Test Portuguese question
            portuguese_response = {
                "question": "Qual é o score médio NPS?",
                "answer": "O score médio NPS é 9.1",
                "sources": [],
                "confidence": "high",
                "method_used": "text2query",
                "execution_time": 2.0,
                "profile": "customized_profile"
            }
            
            mock_engine.ask_question.return_value = portuguese_response
            mock_engine.profile.profile_name = "customized_profile"
            mock_engine.profile.language = "pt-BR"
            mock_get_engine.return_value = mock_engine
            
            # Test Portuguese question
            result = mock_mcp_server.ask_question(
                question="Qual é o score médio NPS?",
                method="auto"
            )
            
            assert "score médio NPS" in result["answer"]
            assert result["profile"] == "customized_profile"
            
            # Test profile info
            profile_result = mock_mcp_server.get_profile_info()
            assert profile_result["language"] == "pt-BR"
    
    @pytest.mark.integration
    def test_nps_data_consistency(self, mock_mcp_server):
        """Test NPS data consistency across tools."""
        with patch.object(mock_mcp_server, '_get_engine') as mock_get_engine:
            mock_engine = Mock()
            
            # Consistent NPS responses
            mock_engine.ask_question.return_value = MOCK_NPS_MCP_ASK_RESPONSE
            mock_engine.search_data.return_value = MOCK_NPS_MCP_SEARCH_RESPONSE
            mock_engine.get_stats.return_value = MOCK_NPS_MCP_STATS_RESPONSE
            mock_engine.profile.profile_name = "customized_profile"
            mock_get_engine.return_value = mock_engine
            
            # Verify consistent profile name across all tools
            ask_result = mock_mcp_server.ask_question("Test question", "auto")
            search_result = mock_mcp_server.search_data("test", 5)
            stats_result = mock_mcp_server.get_stats()
            profile_result = mock_mcp_server.get_profile_info()
            
            assert ask_result["profile"] == "customized_profile"
            assert stats_result["profile"] == "customized_profile"
            assert profile_result["profile_name"] == "customized_profile"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
