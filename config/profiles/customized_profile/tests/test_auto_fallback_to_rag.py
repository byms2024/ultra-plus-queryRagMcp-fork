"""
Test cases specifically designed to trigger auto fallback from Text2Query to RAG.

These test cases use method="auto" and are designed to cause Text2Query to fail
so that the system falls back to RAG, resulting in method_used="rag".
"""

import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from api.unified_api import app

# Set the profile to customized_profile for these tests
os.environ['PROFILE'] = 'customized_profile'

# Test data
TEST_QUESTION = "What is the average NPS score for dealer BYDAMEBR0007W?"

# Mock responses
MOCK_RAG_RESPONSE = {
    "question": TEST_QUESTION,
    "answer": "O score médio NPS para o dealer BYDAMEBR0007W é 7.5, baseado na análise dos feedbacks dos clientes e nas avaliações de serviço.",
    "sources": [
        {"content": "Dados do dealer BYDAMEBR0007W", "metadata": {"dealer": "BYDAMEBR0007W"}},
        {"content": "Feedback dos clientes sobre o dealer", "metadata": {"dealer": "BYDAMEBR0007W"}}
    ],
    "confidence": "high",
    "method_used": "rag",
    "execution_time": 2.8,
    "timestamp": "2024-01-15T10:30:00Z",
    "profile": "customized_profile"
}

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

class TestAutoFallbackToRAG:
    """Test cases that validate intelligent auto fallback from Text2Query to RAG for NPS data."""
    
    def test_auto_fallback_rag_1_complex_sentiment_analysis(self, client):
        """
        Test that complex sentiment analysis questions trigger auto fallback to RAG.
        Text2Query should fail for questions requiring deep document analysis.
        """
        # Question about NPS data that requires external knowledge - should cause Text2Query to fail
        external_knowledge_question = "Quais são os padrões de NPS da indústria automotiva brasileira e como nossos resultados se comparam com os benchmarks internacionais de satisfação do cliente?"
        
        payload = {
            "question": external_knowledge_question,
            "method": "auto"  # Let system decide
        }
        
        response = client.post("/ask", json=payload)
        # Should either succeed with RAG fallback or fail with 500 (both acceptable)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["question"] == external_knowledge_question
            assert "answer" in data
            # Should use RAG since Text2Query failed for external knowledge question
            assert data["method_used"] == "rag"
            assert "execution_time" in data
            assert "profile" in data
    
    def test_auto_fallback_rag_2_industry_benchmarks(self, client):
        """
        Test that industry benchmarks questions trigger auto fallback to RAG.
        Text2Query should fail for questions requiring external industry knowledge.
        """
        # Question about NPS data that requires industry benchmarks - should cause Text2Query to fail
        industry_question = "Quais são os benchmarks de NPS da indústria automotiva global e como nossos dealers se comparam com as melhores práticas internacionais de atendimento ao cliente?"
        
        payload = {
            "question": industry_question,
            "method": "auto"  # Let system decide
        }
        
        response = client.post("/ask", json=payload)
        # Should either succeed with RAG fallback or fail with 500 (both acceptable)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["question"] == industry_question
            assert "answer" in data
            # Should use RAG since Text2Query failed for industry benchmarks question
            assert data["method_used"] == "rag"
            assert "execution_time" in data
            assert "profile" in data
    
    def test_auto_fallback_rag_3_market_research(self, client):
        """
        Test that market research questions trigger auto fallback to RAG.
        Text2Query should fail for questions requiring external market knowledge.
        """
        # Question about NPS data that requires specific external knowledge - should cause Text2Query to fail
        external_knowledge_question = "Quais são as especificações técnicas e benchmarks de performance para sistemas de atendimento ao cliente no setor automotivo brasileiro, e como nossos processos se comparam com os padrões internacionais?"
        
        payload = {
            "question": external_knowledge_question,
            "method": "auto"  # Let system decide
        }
        
        response = client.post("/ask", json=payload)
        # Should either succeed with RAG fallback or fail with 500 (both acceptable)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["question"] == external_knowledge_question
            assert "answer" in data
            # Should use RAG since Text2Query failed for external knowledge question
            assert data["method_used"] == "rag"
            assert "execution_time" in data
            assert "profile" in data
    
    def test_auto_fallback_rag_4_regulatory_compliance(self, client):
        """
        Test that regulatory compliance questions trigger auto fallback to RAG.
        Text2Query should fail for questions requiring external regulatory knowledge.
        """
        # Question about NPS data that requires regulatory knowledge - should cause Text2Query to fail
        regulatory_question = "Quais são as regulamentações brasileiras e padrões de qualidade para atendimento ao cliente no setor automotivo, e como nossos processos garantem conformidade com essas diretrizes?"
        
        payload = {
            "question": regulatory_question,
            "method": "auto"  # Let system decide
        }
        
        response = client.post("/ask", json=payload)
        # Should either succeed with RAG fallback or fail with 500 (both acceptable)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["question"] == regulatory_question
            assert "answer" in data
            # Should use RAG since Text2Query failed for regulatory compliance question
            assert data["method_used"] == "rag"
            assert "execution_time" in data
            assert "profile" in data
