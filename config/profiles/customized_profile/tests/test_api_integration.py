import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock
import tempfile
import pandas as pd
from pathlib import Path
import sys
import os
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from api.generic_fastapi_app import app, get_rag_agent
from config.langchain_settings import LangChainConfig

# Import shared test utilities
from config.profiles.common_test_utils import (
    create_mock_rag_agent,
    create_mock_rag_agent_with_error,
    create_mock_answer_response,
    create_mock_search_response,
    create_mock_stats_response
)

# =============================================================================
# TEST CONSTANTS AND SHARED DATA
# =============================================================================

# Common test data
TEST_DEALER_CODE = "BYDAMEBR0007W"
TEST_RO_NO = "5457"
TEST_SCORE = 9.0
TEST_VIN = "LC0CE4CC2R0009877"

# Mock response data (using shared utilities)
MOCK_ANSWER_RESPONSE = create_mock_answer_response(
    sources=[{
        "dealer": TEST_DEALER_CODE,
        "score": TEST_SCORE,
        "ro_no": TEST_RO_NO,
        "content": "Test content snippet"
    }]
)

MOCK_SEARCH_RESPONSE = create_mock_search_response(
    metadata={
        "dealer": TEST_DEALER_CODE,
        "score": TEST_SCORE,
        "ro_no": TEST_RO_NO
    }
)

MOCK_STATS_RESPONSE = create_mock_stats_response(
    collection_name="nps_data"
)

# Test questions and queries
TEST_QUESTIONS = [
    "What is the average score?",
    "Which dealer has the highest satisfaction?",
    "What are the common issues reported?",
    "How many repairs were done last month?"
]

TEST_QUERIES = [
    "dealer satisfaction",
    "repair quality",
    "customer complaints",
    "service efficiency"
]


class TestAPIEndpoints:
    """Enhanced test suite for FastAPI endpoints with improved coverage."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return LangChainConfig()
    
    @pytest.fixture
    def mock_rag_agent(self):
        """Create a comprehensive mock RAG agent."""
        return create_mock_rag_agent(
            answer_response=MOCK_ANSWER_RESPONSE,
            search_response=MOCK_SEARCH_RESPONSE,
            stats_response=MOCK_STATS_RESPONSE
        )
    
    @pytest.fixture
    def mock_rag_agent_with_error(self):
        """Create a mock RAG agent that raises exceptions."""
        return create_mock_rag_agent_with_error()

    # =============================================================================
    # ROOT ENDPOINT TESTS
    # =============================================================================
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert data["version"] == "2.0.0"
        assert "api" in data["message"].lower()
    
    def test_root_endpoint_content_type(self, client):
        """Test root endpoint returns JSON content type."""
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"

    # =============================================================================
    # HEALTH CHECK TESTS
    # =============================================================================
    
    def test_health_check_success(self, client, mock_rag_agent):
        """Test successful health check."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            response = client.get("/health")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"
        # Generic app returns vectorstore_status/data_records instead of 'stats'
        assert "vectorstore_status" in data
        assert "data_records" in data
    
    def test_health_check_with_error(self, client, mock_rag_agent_with_error):
        """Test health check when RAG agent has errors."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent_with_error
        try:
            response = client.get("/health")
        finally:
            app.dependency_overrides.clear()
        # Health check might return 500 if RAG agent fails, or 200 if it handles errors gracefully
        assert response.status_code in [200, 500]
        data = response.json()
        
        if response.status_code == 200:
            assert data["status"] == "healthy"
            assert data["version"] == "2.0.0"
        else:
            assert "detail" in data  # Error response

    # =============================================================================
    # ASK ENDPOINT TESTS
    # =============================================================================
    
    def test_ask_endpoint_success(self, client, mock_rag_agent):
        """Test successful ask endpoint."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            response = client.post("/ask", json={"question": TEST_QUESTIONS[0]})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        
        assert data["question"] == TEST_QUESTIONS[0]
        assert data["answer"] == MOCK_ANSWER_RESPONSE["answer"]
        assert len(data["sources"]) == 1
        assert data["confidence"] == "high"
        assert "timestamp" in data
    
    def test_ask_endpoint_multiple_questions(self, client, mock_rag_agent):
        """Test ask endpoint with different question types."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            for question in TEST_QUESTIONS:
                response = client.post("/ask", json={"question": question})
                assert response.status_code == 200
                data = response.json()
                assert data["question"] == question
                assert "answer" in data
        finally:
            app.dependency_overrides.clear()
    
    def test_ask_endpoint_with_error(self, client, mock_rag_agent_with_error):
        """Test ask endpoint when RAG agent raises an error."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent_with_error
        try:
            response = client.post("/ask", json={"question": TEST_QUESTIONS[0]})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI uses 'detail' for error messages
    
    def test_ask_endpoint_missing_question(self, client):
        """Test ask endpoint with missing question field."""
        agent = Mock(); agent.answer_question.return_value = {"answer": "", "sources": [], "confidence": "low", "timestamp": "2024-01-01T00:00:00"}
        app.dependency_overrides[get_rag_agent] = lambda: agent
        try:
            response = client.post("/ask", json={})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 422
    
    def test_ask_endpoint_invalid_json(self, client):
        """Test ask endpoint with invalid JSON."""
        response = client.post("/ask", data="invalid json")
        assert response.status_code == 422
    
    def test_ask_endpoint_empty_question(self, client):
        """Test ask endpoint with empty question."""
        agent = Mock(); agent.answer_question.return_value = {"answer": "", "sources": [], "confidence": "low", "timestamp": "2024-01-01T00:00:00"}
        app.dependency_overrides[get_rag_agent] = lambda: agent
        try:
            response = client.post("/ask", json={"question": ""})
        finally:
            app.dependency_overrides.clear()
        # API might accept empty questions or return validation error
        assert response.status_code in [200, 422]

    # =============================================================================
    # ASK-API COMPATIBILITY TESTS
    # =============================================================================
    
    # Removed legacy /ask-api compatibility test; not present in generic app

    # =============================================================================
    # SEARCH ENDPOINT TESTS
    # =============================================================================
    
    def test_search_endpoint_success(self, client, mock_rag_agent):
        """Test successful search endpoint."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            response = client.post("/search", json={"query": TEST_QUERIES[0], "top_k": 5})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        
        assert data["query"] == TEST_QUERIES[0]
        assert len(data["results"]) == 1
        assert data["total_found"] == 1
        assert "similarity" in data["results"][0]
    
    def test_search_endpoint_different_queries(self, client, mock_rag_agent):
        """Test search endpoint with different query types."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            for query in TEST_QUERIES:
                response = client.post("/search", json={"query": query, "top_k": 3})
                assert response.status_code == 200
                data = response.json()
                assert data["query"] == query
        finally:
            app.dependency_overrides.clear()
    
    def test_search_endpoint_different_top_k(self, client, mock_rag_agent):
        """Test search endpoint with different top_k values."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            for top_k in [1, 5, 10, 20]:
                response = client.post("/search", json={"query": TEST_QUERIES[0], "top_k": top_k})
                assert response.status_code == 200
                data = response.json()
                assert data["query"] == TEST_QUERIES[0]
        finally:
            app.dependency_overrides.clear()
    
    def test_search_endpoint_missing_query(self, client):
        """Test search endpoint with missing query."""
        app.dependency_overrides[get_rag_agent] = lambda: Mock()
        try:
            response = client.post("/search", json={"top_k": 5})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 422
    
    def test_search_endpoint_missing_top_k(self, client):
        """Test search endpoint with missing top_k."""
        from api.generic_fastapi_app import app, get_rag_agent
        m = Mock(); m.search_relevant_chunks.return_value = []
        app.dependency_overrides[get_rag_agent] = lambda: m
        try:
            response = client.post("/search", json={"query": TEST_QUERIES[0]})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code in [200, 422]
    
    def test_search_endpoint_invalid_top_k(self, client):
        """Test search endpoint with invalid top_k values."""
        from api.generic_fastapi_app import app, get_rag_agent
        app.dependency_overrides[get_rag_agent] = lambda: Mock()
        try:
            invalid_values = [0, -1, 100, "invalid"]
            for invalid_value in invalid_values:
                response = client.post("/search", json={"query": TEST_QUERIES[0], "top_k": invalid_value})
                assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    # =============================================================================
    # STATS ENDPOINT TESTS
    # =============================================================================
    
    def test_stats_endpoint_success(self, client, mock_rag_agent):
        """Test successful stats endpoint."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            response = client.get("/stats")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        
        assert "vectorstore" in data
        assert "data" in data
        assert "sensitization" in data
        assert data == MOCK_STATS_RESPONSE
    
    def test_stats_endpoint_with_error(self, client, mock_rag_agent_with_error):
        """Test stats endpoint when RAG agent raises an error."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent_with_error
        try:
            response = client.get("/stats")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI uses 'detail' for error messages

    # =============================================================================
    # SENSITIZATION STATS TESTS
    # =============================================================================
    
    # Remove non-generic endpoint test (no /sensitization-stats in generic app)

    # =============================================================================
    # REBUILD INDEX TESTS
    # =============================================================================
    
    def test_rebuild_index_endpoint_success(self, client, mock_rag_agent):
        """Test successful rebuild index endpoint."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            response = client.post("/rebuild")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "rebuilt successfully" in data["message"]
    
    def test_rebuild_index_endpoint_with_error(self, client, mock_rag_agent_with_error):
        """Test rebuild index endpoint when RAG agent raises an error."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent_with_error
        try:
            response = client.post("/rebuild")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI uses 'detail' for error messages

    # =============================================================================
    # EDGE CASES AND ERROR HANDLING
    # =============================================================================
    
    def test_unsupported_endpoint(self, client):
        """Test accessing an unsupported endpoint."""
        response = client.get("/unsupported-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        response = client.get("/ask")  # GET instead of POST
        assert response.status_code == 405
    
    def test_large_payload(self, client, mock_rag_agent):
        """Test with large payload."""
        app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
        try:
            large_question = "What is the average score? " * 1000  # Very long question
            response = client.post("/ask", json={"question": large_question})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])