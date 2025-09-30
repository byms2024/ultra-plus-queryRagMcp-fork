import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config.langchain_settings import LangChainConfig
from config.profiles.profile_factory import ProfileFactory
from rag.generic_data_processor import GenericDataProcessor, DataSchema
from rag.generic_vector_store import GenericVectorStore
from rag.generic_rag_agent import GenericRAGAgent

# Import shared test utilities
from config.profiles.common_test_utils import (
    create_mock_llm,
    create_mock_embeddings,
    create_mock_rag_agent,
    create_test_csv,
    cleanup_temp_file,
    BaseRAGAgentTest,
    BaseVectorStoreTest,
    BaseDataProcessorTest,
    MOCK_LLM_RESPONSE,
    MOCK_EMBEDDING
)

# =============================================================================
# TEST CONSTANTS AND SHARED DATA
# =============================================================================

# BR Profile specific test data
BR_TEST_DATA = {
    "dealer_code": "BYDAMEBR0007W",
    "ro_no": "5457",
    "score": 9.0,
    "vin": "LC0CE4CC2R0009877",
    "repair_type": "Other Repair",
    "trouble_desc": "Test issue description",
    "check_result": "Test check result"
}

# CSV headers for BR profile
BR_CSV_HEADERS = [
    "ID", "DEALER_CODE", "SUB_DEALER_CODE", "REPAIR_STORE_CODE", "RO_NO", "SCORE",
    "SERVICE_ATTITUDE", "ENVIRONMENT", "EFFICIENCY", "EFFECTIVENESS", "PARTS_AVAILABILITY",
    "OTHERS", "SUBMIT_DATE", "CREATE_DATE", "UPDATE_DATE", "IS_ANON", "QUE_THREE_REASON",
    "ORDER_CREATE_DATE", "ORDER_LAST_BALANCE_DATE", "TROUBLE_DESC", "CHECK_RESULT",
    "DELIVER_PROBLEM", "VIN", "REPAIR_TYPE_NAME", "SERVICE_ATTITUDE", "ENVIRONMENT",
    "EFFICIENCY", "EFFECTIVENESS", "PARTS_AVAILABILITY", "OTHERS", "OTHERS_REASON"
]

# Sample CSV row data
SAMPLE_CSV_ROW = [
    "C0125030811193253828", BR_TEST_DATA["dealer_code"], "", "", BR_TEST_DATA["ro_no"],
    str(BR_TEST_DATA["score"]), "Y", "", "", "", "", "", "2025-03-08 11:19:32.000",
    "2025-03-09 00:01:22.000", "2025-03-11 06:51:59.000", "否", "",
    "2025-02-25 00:00:00.000", "2025-02-26 23:59:59.000", BR_TEST_DATA["trouble_desc"],
    BR_TEST_DATA["check_result"], "", BR_TEST_DATA["vin"], BR_TEST_DATA["repair_type"],
    "Y", "", "", "", "", "", ""
]

# Mock responses (using shared constants)
MOCK_DOCUMENT_CONTENT = f"Ordem {BR_TEST_DATA['ro_no']} do dealer {BR_TEST_DATA['dealer_code']} com score {BR_TEST_DATA['score']}"

class TestGenericDataProcessor:
    """Test the generic data processor with BR profile data."""

    def create_test_csv(self, rows=None, headers=None):
        """Helper to create a temporary CSV file."""
        if headers is None:
            headers = BR_CSV_HEADERS
        if rows is None:
            rows = [SAMPLE_CSV_ROW]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(",".join(f'"{header}"' for header in headers) + "\n")
            for row in rows:
                f.write(",".join(f'"{str(cell)}"' for cell in row) + "\n")
            return f.name

    def get_br_data_schema(self):
        """Get BR profile data schema."""
        return DataSchema(
            required_columns=BR_CSV_HEADERS,
            sensitive_columns=['DEALER_CODE', 'SUB_DEALER_CODE', 'VIN'],
            date_columns=['CREATE_DATE', 'SUBMIT_DATE', 'UPDATE_DATE'],
            text_columns=['TROUBLE_DESC', 'CHECK_RESULT', 'OTHERS_REASON'],
            metadata_columns=['RO_NO', 'SCORE', 'SERVICE_ATTITUDE', 'ENVIRONMENT', 'EFFICIENCY', 'EFFECTIVENESS', 'PARTS_AVAILABILITY', 'OTHERS', 'REPAIR_TYPE_NAME'],
            id_column='RO_NO',
            score_column='SCORE'
        )

    def test_data_processor_initialization(self):
        """Test data processor initialization."""
        temp_csv = self.create_test_csv()
        try:
            schema = self.get_br_data_schema()
            processor = GenericDataProcessor(temp_csv, schema)
            assert processor.csv_path == temp_csv
            assert processor.schema == schema
            assert "RO_NO" in processor.schema.required_columns
            assert "SCORE" in processor.schema.required_columns
        finally:
            cleanup_temp_file(temp_csv)

    def test_data_processor_initialization_invalid_file(self):
        """Test data processor initialization with invalid file."""
        schema = self.get_br_data_schema()
        processor = GenericDataProcessor("nonexistent_file.csv", schema)
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            processor.load_and_process_data()

    def test_data_loading_and_processing(self):
        """Test data loading and processing with BR profile data."""
        temp_csv = self.create_test_csv()
        
        try:
            schema = self.get_br_data_schema()
            processor = GenericDataProcessor(temp_csv, schema)
            df = processor.load_and_process_data()
            
            assert len(df) == 1
            assert str(df.iloc[0]['RO_NO']) == BR_TEST_DATA["ro_no"]
            assert df.iloc[0]['SCORE'] == BR_TEST_DATA["score"]
            # Dealer code will be sensitized
            assert 'DEALER_' in df.iloc[0]['DEALER_CODE']
            # VIN will be sensitized
            assert 'VIN_' in df.iloc[0]['VIN']
            
            # Check sensitization
            assert 'DEALER_' in df.iloc[0]['DEALER_CODE']  # Should be sensitized
        finally:
            cleanup_temp_file(temp_csv)

    def test_data_processing_multiple_rows(self):
        """Test data processing with multiple rows."""
        rows = [
            SAMPLE_CSV_ROW,
            ["C0125030811193253829", "BYDAMEBR0008W", "", "", "5458", "8", "Y", "", "", "", "", "", 
             "2025-03-08 11:19:32.000", "2025-03-09 00:01:22.000", "2025-03-11 06:51:59.000", "否", "",
             "2025-02-25 00:00:00.000", "2025-02-26 23:59:59.000", "Another issue", "Another result", 
             "", "LC0CE4CC2R0009878", "Other Repair", "Y", "", "", "", "", "", ""]
        ]
        temp_csv = self.create_test_csv(rows=rows)
        
        try:
            schema = self.get_br_data_schema()
            processor = GenericDataProcessor(temp_csv, schema)
            df = processor.load_and_process_data()
            
            assert len(df) == 2
            assert str(df.iloc[0]['RO_NO']) == BR_TEST_DATA["ro_no"]
            assert str(df.iloc[1]['RO_NO']) == "5458"
            
            # Check that both dealer codes are sensitized
            assert 'DEALER_' in df.iloc[0]['DEALER_CODE']
            assert 'DEALER_' in df.iloc[1]['DEALER_CODE']
            # And they should be different
            assert df.iloc[0]['DEALER_CODE'] != df.iloc[1]['DEALER_CODE']
        finally:
            cleanup_temp_file(temp_csv)

    def test_document_creation(self):
        """Test document creation from processed data."""
        temp_csv = self.create_test_csv()
        
        try:
            schema = self.get_br_data_schema()
            processor = GenericDataProcessor(temp_csv, schema)
            df = processor.load_and_process_data()
            documents = processor.create_documents()
            
            assert len(documents) == 1
            document = documents[0]
            
            # Check document content contains text fields
            assert BR_TEST_DATA["trouble_desc"] in document.page_content
            assert BR_TEST_DATA["check_result"] in document.page_content
            
            # Check metadata
            assert 'ro_no' in document.metadata
            assert 'score' in document.metadata
            assert document.metadata['ro_no'] == BR_TEST_DATA["ro_no"]
            assert document.metadata['score'] == BR_TEST_DATA["score"]
            
            # Check sensitized fields in metadata
            if 'dealer' in document.metadata:
                assert 'DEALER_' in document.metadata['dealer']
            if 'vin' in document.metadata:
                assert 'VIN_' in document.metadata['vin']
        finally:
            cleanup_temp_file(temp_csv)

    def test_document_creation_multiple_documents(self):
        """Test document creation with multiple records."""
        rows = [
            SAMPLE_CSV_ROW,
            ["C0125030811193253829", "BYDAMEBR0008W", "", "", "5458", "8", "Y", "", "", "", "", "", 
             "2025-03-08 11:19:32.000", "2025-03-09 00:01:22.000", "2025-03-11 06:51:59.000", "否", "",
             "2025-02-25 00:00:00.000", "2025-02-26 23:59:59.000", "Another issue", "Another result", 
             "", "LC0CE4CC2R0009878", "Other Repair", "Y", "", "", "", "", "", ""]
        ]
        temp_csv = self.create_test_csv(rows=rows)
        
        try:
            schema = self.get_br_data_schema()
            processor = GenericDataProcessor(temp_csv, schema)
            df = processor.load_and_process_data()
            documents = processor.create_documents()
            
            assert len(documents) == 2
            
            # Check both documents have different content
            assert documents[0].page_content != documents[1].page_content
            assert documents[0].metadata['ro_no'] != documents[1].metadata['ro_no']
        finally:
            cleanup_temp_file(temp_csv)

    def test_chunk_creation(self):
        """Test chunk creation from documents."""
        temp_csv = self.create_test_csv()
        
        try:
            schema = self.get_br_data_schema()
            processor = GenericDataProcessor(temp_csv, schema)
            df = processor.load_and_process_data()
            documents = processor.create_documents()
            chunks = processor.create_chunks(documents)
            
            assert len(chunks) >= 1  # At least one chunk
            assert all(isinstance(chunk, type(documents[0])) for chunk in chunks)
        finally:
            cleanup_temp_file(temp_csv)

    def test_sensitive_mapping(self):
        """Test sensitive data mapping functionality."""
        temp_csv = self.create_test_csv()
        
        try:
            schema = self.get_br_data_schema()
            processor = GenericDataProcessor(temp_csv, schema)
            df = processor.load_and_process_data()
            mapping = processor.get_sensitive_mapping()
            
            # Should have mappings for sensitive columns
            assert len(mapping) > 0
            # Original values should be in mapping keys
            assert BR_TEST_DATA["dealer_code"] in mapping
            assert BR_TEST_DATA["vin"] in mapping
        finally:
            cleanup_temp_file(temp_csv)

    def test_get_stats(self):
        """Test getting processing statistics."""
        temp_csv = self.create_test_csv()
        
        try:
            schema = self.get_br_data_schema()
            processor = GenericDataProcessor(temp_csv, schema)
            df = processor.load_and_process_data()
            stats = processor.get_stats()
            
            assert stats["total_records"] == 1
            assert stats["sensitive_mappings"] > 0
            assert "score_stats" in stats
            assert stats["score_stats"]["mean"] == BR_TEST_DATA["score"]
        finally:
            cleanup_temp_file(temp_csv)

class TestGenericVectorStore(BaseVectorStoreTest):
    """Test the generic vector store."""

    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        self._test_vector_store_initialization(GenericVectorStore, "/tmp/test", "test_collection")

    def test_vector_store_initialization_with_custom_collection(self):
        """Test vector store initialization with custom collection name."""
        self._test_vector_store_initialization_with_custom_collection(GenericVectorStore, "/tmp/test")

    def test_get_stats_not_initialized(self):
        """Test getting stats when vector store is not initialized."""
        self._test_get_stats_not_initialized(GenericVectorStore, "/tmp/test")

class TestGenericRAGAgent(BaseRAGAgentTest):
    """Test the generic RAG agent with BR profile data."""
    
    def get_br_data_schema(self):
        """Get BR profile data schema."""
        return DataSchema(
            required_columns=BR_CSV_HEADERS,
            sensitive_columns=['DEALER_CODE', 'SUB_DEALER_CODE', 'VIN'],
            date_columns=['CREATE_DATE', 'SUBMIT_DATE', 'UPDATE_DATE'],
            text_columns=['TROUBLE_DESC', 'CHECK_RESULT', 'OTHERS_REASON'],
            metadata_columns=['RO_NO', 'SCORE', 'SERVICE_ATTITUDE', 'ENVIRONMENT', 'EFFICIENCY', 'EFFECTIVENESS', 'PARTS_AVAILABILITY', 'OTHERS', 'REPAIR_TYPE_NAME'],
            id_column='RO_NO',
            score_column='SCORE'
        )

    def create_test_csv(self):
        """Helper to create a temporary CSV file."""
        return create_test_csv(BR_CSV_HEADERS, [SAMPLE_CSV_ROW])

    @pytest.mark.slow
    @pytest.mark.external
    @patch('config.providers.registry.ChatGoogleGenerativeAI')
    @patch('config.providers.registry.GoogleGenerativeAIEmbeddings')
    def test_rag_agent_initialization_success(self, mock_embeddings, mock_llm):
        if os.getenv('RUN_EXTERNAL') != '1':
            pytest.skip('external smoke test disabled (set RUN_EXTERNAL=1 to enable)')
        """Test successful RAG agent initialization."""
        temp_csv = self.create_test_csv()
        
        try:
            mock_llm.return_value = create_mock_llm()
            mock_embeddings.return_value = create_mock_embeddings()
            
            # Create a mock config object
            config = Mock()
            config.csv_file = temp_csv
            config.vector_store_path = '/tmp/test_vectorstore'
            config.generation_model = 'gemini-pro'
            config.embedding_model = 'embedding-001'
            config.temperature = 0.7
            config.max_tokens = 1000
            config.sample_size = None
            
            schema = self.get_br_data_schema()
            agent = GenericRAGAgent(config, schema, "test_collection")
            assert agent.config == config
            assert agent.data_schema == schema
            assert agent.collection_name == "test_collection"
            assert agent.llm is not None
            assert agent.embeddings is not None
            assert agent.data_processor is not None
            assert agent.vectorstore is not None
        finally:
            cleanup_temp_file(temp_csv)

    @pytest.mark.skip(reason="Complex mocking issues with LangChain validation")
    @pytest.mark.slow
    @pytest.mark.external
    @patch('rag.generic_rag_agent.ChatGoogleGenerativeAI')
    @patch('rag.generic_rag_agent.GoogleGenerativeAIEmbeddings')
    def test_rag_agent_answer_question(self, mock_embeddings, mock_llm):
        if os.getenv('RUN_EXTERNAL') != '1':
            pytest.skip('external smoke test disabled (set RUN_EXTERNAL=1 to enable)')
        """Test RAG agent question answering."""
        temp_csv = self.create_test_csv()
        
        try:
            mock_llm.return_value = create_mock_llm()
            mock_embeddings.return_value = create_mock_embeddings()
            
            # Create a mock config object
            config = Mock()
            config.csv_file = temp_csv
            config.vector_store_path = '/tmp/test_vectorstore'
            config.generation_model = 'gemini-pro'
            config.embedding_model = 'embedding-001'
            config.temperature = 0.7
            config.max_tokens = 1000
            config.sample_size = None
            
            schema = self.get_br_data_schema()
            agent = GenericRAGAgent(config, schema, "test_collection")
            
            # Mock the QA chain
            mock_qa_chain = Mock()
            mock_qa_chain.return_value = {
                "result": "Test answer",
                "source_documents": []
            }
            agent.qa_chain = mock_qa_chain
            
            result = agent.answer_question("What is the average score?")
            
            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result
            assert "timestamp" in result
        finally:
            cleanup_temp_file(temp_csv)

    @pytest.mark.skip(reason="Complex mocking issues with LangChain validation")
    @pytest.mark.slow
    @pytest.mark.external
    @patch('rag.generic_rag_agent.ChatGoogleGenerativeAI')
    @patch('rag.generic_rag_agent.GoogleGenerativeAIEmbeddings')
    def test_rag_agent_search_relevant_chunks(self, mock_embeddings, mock_llm):
        if os.getenv('RUN_EXTERNAL') != '1':
            pytest.skip('external smoke test disabled (set RUN_EXTERNAL=1 to enable)')
        """Test RAG agent chunk searching."""
        temp_csv = self.create_test_csv()
        
        try:
            mock_llm.return_value = create_mock_llm()
            mock_embeddings.return_value = create_mock_embeddings()
            
            # Create a mock config object
            config = Mock()
            config.csv_file = temp_csv
            config.vector_store_path = '/tmp/test_vectorstore'
            config.generation_model = 'gemini-pro'
            config.embedding_model = 'embedding-001'
            config.temperature = 0.7
            config.max_tokens = 1000
            config.sample_size = None
            
            schema = self.get_br_data_schema()
            agent = GenericRAGAgent(config, schema, "test_collection")
            
            # Mock the vector store
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search_with_score = Mock(return_value=[])
            agent.vectorstore = mock_vectorstore
            
            results = agent.search_relevant_chunks("test query", 5)
            
            assert isinstance(results, list)
        finally:
            cleanup_temp_file(temp_csv)

    @pytest.mark.skip(reason="Complex mocking issues with LangChain validation")
    @pytest.mark.slow
    @pytest.mark.external
    @patch('rag.generic_rag_agent.ChatGoogleGenerativeAI')
    @patch('rag.generic_rag_agent.GoogleGenerativeAIEmbeddings')
    def test_rag_agent_get_stats(self, mock_embeddings, mock_llm):
        if os.getenv('RUN_EXTERNAL') != '1':
            pytest.skip('external smoke test disabled (set RUN_EXTERNAL=1 to enable)')
        """Test RAG agent statistics."""
        temp_csv = self.create_test_csv()
        
        try:
            mock_llm.return_value = create_mock_llm()
            mock_embeddings.return_value = create_mock_embeddings()
            
            # Create a mock config object
            config = Mock()
            config.csv_file = temp_csv
            config.vector_store_path = '/tmp/test_vectorstore'
            config.generation_model = 'gemini-pro'
            config.embedding_model = 'embedding-001'
            config.temperature = 0.7
            config.max_tokens = 1000
            config.sample_size = None
            
            schema = self.get_br_data_schema()
            agent = GenericRAGAgent(config, schema, "test_collection")
            
            stats = agent.get_stats()
            
            assert "vectorstore" in stats
            assert "data" in stats
            assert "sensitization" in stats
        finally:
            cleanup_temp_file(temp_csv)
