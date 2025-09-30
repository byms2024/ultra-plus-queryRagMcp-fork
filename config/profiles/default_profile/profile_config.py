#!/usr/bin/env python3
"""
Default profile for generic data processing.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from ..base_profile import BaseProfile, ColumnDefinition, SensitizationRule, DocumentTemplate
from core.rag.generic_data_processor import DataSchema
from reports.generic_report_builder import ReportConfig

# =============================================================================
# CONSTANTS - Profile-specific configuration overrides
# Uncomment and modify these to override the default values from base_config.py
# =============================================================================

class CONSTANTS:
    """Profile-specific configuration constants that override base defaults."""
    
    # LLM Configuration Overrides
    # GENERATION_MODEL = "gemini-2.5-flash"  # Override default generation model
    # EMBEDDING_MODEL = "text-embedding-004"  # Override default embedding model
    # TEMPERATURE = 0.2  # Override default temperature
    # MAX_TOKENS = 2048  # Override default max tokens
    
    # Vector Store Configuration Overrides
    # VECTOR_STORE_TYPE = "chroma"  # Override default vector store type (chroma/faiss)
    # CHUNK_SIZE = 1000  # Override default chunk size
    # CHUNK_OVERLAP = 200  # Override default chunk overlap
    
    # RAG Configuration Overrides
    # TOP_K = 50  # Override default top_k for retrieval
    # MAX_ITERATIONS = 10  # Override default max iterations
    
    # API Configuration Overrides
    # API_PORT = 8000  # Override default API port
    # MCP_PORT = 7800  # Override default MCP port
    
    # Data Configuration Overrides
    # SAMPLE_SIZE = None  # Override default sample size (None for no limit)
    
    # Logging Configuration Overrides
    # LOG_LEVEL = "INFO"  # Override default log level
    # LOG_TO_FILE = True  # Override default log to file setting
    # LOG_TO_CONSOLE = True  # Override default log to console setting
    
    # LangSmith Configuration Overrides
    # LANGSMITH_API_KEY = None  # Override default LangSmith API key
    # LANGSMITH_PROJECT = "default-profile-rag"  # Override default LangSmith project
    # ENABLE_TRACING = False  # Override default tracing setting
    
    pass  # Remove this line when adding actual overrides


class DefaultProfile(BaseProfile):
    """Default profile for generic data processing."""
    
    def _initialize_profile(self):
        """Initialize default profile settings."""
        # Profile identification
        self.profile_name = "default_profile"
        
        # Language settings
        self.language = "en-US"
        self.locale = "en_US"
        
        # File paths
        self.data_file_path = str(Path(__file__).parent / "test_data" / "fridge_sales_with_rating.csv")
        self.test_data_path = str(Path(__file__).parent / "test_data")
        
        # Define required columns for fridge sales data
        self.required_columns = [
            'ID', 'CUSTOMER_ID', 'FRIDGE_MODEL', 'BRAND', 'CAPACITY_LITERS', 
            'PRICE', 'SALES_DATE', 'STORE_NAME', 'STORE_ADDRESS', 'CUSTOMER_FEEDBACK', 'FEEDBACK_RATING'
        ]
        
        # Define column definitions
        self.column_definitions = {
            'ID': ColumnDefinition('ID', 'string', True, "Product ID"),
            'CUSTOMER_ID': ColumnDefinition('CUSTOMER_ID', 'string', True, "Customer ID", sensitive=True),
            'FRIDGE_MODEL': ColumnDefinition('FRIDGE_MODEL', 'string', True, "Fridge Model"),
            'BRAND': ColumnDefinition('BRAND', 'string', True, "Brand"),
            'CAPACITY_LITERS': ColumnDefinition('CAPACITY_LITERS', 'int', True, "Capacity in Liters"),
            'PRICE': ColumnDefinition('PRICE', 'float', True, "Price"),
            'SALES_DATE': ColumnDefinition('SALES_DATE', 'datetime', True, "Sales Date"),
            'STORE_NAME': ColumnDefinition('STORE_NAME', 'string', True, "Store Name"),
            'STORE_ADDRESS': ColumnDefinition('STORE_ADDRESS', 'string', True, "Store Address"),
            'CUSTOMER_FEEDBACK': ColumnDefinition('CUSTOMER_FEEDBACK', 'string', True, "Customer Feedback", text_field=True),
            'FEEDBACK_RATING': ColumnDefinition('FEEDBACK_RATING', 'string', True, "Feedback Rating")
        }
        
        # Define text columns
        self.text_columns = ['CUSTOMER_FEEDBACK']
        
        # Define sensitive columns
        self.sensitive_columns = ['CUSTOMER_ID']
        
        # Define date columns
        self.date_columns = ['SALES_DATE']
        
        # Define sensitization rules (none by default)
        self.sensitization_rules = {}
        
        # Define document template (generic English)
        self.document_template = DocumentTemplate(
            template=(
                "Record {id} has a score of {score}. "
                "Date: {date}. "
                "Description: {description}. "
                "Category: {category}."
            ),
            metadata_fields=[
                "id", "score", "date", "category"
            ]
        )
    
    def get_prompt_template(self) -> str:
        """Get English prompt template for the RAG system."""
        return """
        You are a data analysis assistant.
        Your role is to answer questions based on the provided data.

        Context information:
        {context}

        Question: {question}

        Instructions:
        1. Use only the provided context information to answer the question.
        2. Be specific and cite relevant data when appropriate.
        3. If there is not enough information to answer, state that clearly.
        4. Always answer in English.
        5. When mentioning specific data, be precise.

        Answer:
        """
    
    def get_report_config(self) -> Dict[str, Any]:
        """Get English report generation configuration."""
        return {
            "title": "Data Report",
            "max_rows": 500,
            "date_column": "date",
            "score_column": "score",
            "dealer_column": "category",
            "repair_type_column": "category",
            "language": "en-US"
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize data according to default profile rules."""
        # Clean text columns
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Clean score column
        if 'score' in df.columns:
            score_num = pd.to_numeric(df['score'], errors='coerce')
            df['score'] = score_num.fillna(0).astype(float)
        
        # Clean date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
    
    def create_document_metadata(self, row_data: Dict[str, Any], row_index: int) -> Dict[str, Any]:
        """Create document metadata with English field names."""
        metadata = {"row": row_index}
        
        # Map to English metadata field names
        field_mapping = {
            "id": "id",
            "score": "score", 
            "date": "date",
            "category": "category"
        }
        
        for metadata_field, csv_field in field_mapping.items():
            if csv_field in row_data:
                metadata[metadata_field] = row_data[csv_field]
        
        return metadata
    
    def get_test_directory(self) -> str:
        """Get the path to test directory for default profile."""
        return str(Path(__file__).parent / "tests")
    
    def get_data_schema(self) -> DataSchema:
        """Get the data schema for default profile (fridge sales)."""
        return DataSchema(
            required_columns=self.required_columns,
            sensitive_columns=self.sensitive_columns,
            date_columns=['SALES_DATE'],
            text_columns=self.text_columns,
            metadata_columns=['ID', 'FRIDGE_MODEL', 'BRAND', 'CAPACITY_LITERS', 'PRICE', 'STORE_NAME', 'STORE_ADDRESS'],
            id_column='ID',
            score_column='PRICE'  # Use price as the "score" for fridge sales
        )
    
    def get_report_config(self) -> ReportConfig:
        """Get the report configuration for default profile (fridge sales)."""
        return ReportConfig(
            title="Fridge Sales Report",
            date_columns=['SALES_DATE'],
            score_columns=['PRICE'],
            filter_columns=['STORE_NAME', 'BRAND', 'FRIDGE_MODEL'],
            display_columns=['ID', 'CUSTOMER_ID', 'FRIDGE_MODEL', 'BRAND', 'CAPACITY_LITERS', 'PRICE', 'SALES_DATE', 'STORE_NAME', 'CUSTOMER_FEEDBACK'],
            max_rows=500
        )
    
    def get_censoring_mappings(self) -> Dict[str, str]:
        """Get censoring mappings for sensitive data."""
        return {}
    
    def get_schema_hints(self, sample_data: str) -> str:
        """Get schema hints for the data."""
        return f"""
Schema hints for fridge sales data:
- ID: Unique identifier for each record
- CUSTOMER_ID: Customer identifier (sensitive)
- FRIDGE_MODEL: Model name of the fridge
- BRAND: Brand of the fridge (e.g., Samsung, GE, KitchenAid)
- CAPACITY_LITERS: Storage capacity in liters
- PRICE: Price of the fridge in dollars
- SALES_DATE: Date when the sale occurred
- STORE_NAME: Name of the store where the sale occurred
- STORE_ADDRESS: Address of the store
- CUSTOMER_FEEDBACK: Customer feedback text
"""
    
    def get_llm_system_prompt(self) -> str:
        """Get system prompt for LLM interactions."""
        return """
You are a data analysis assistant for fridge sales data.
Your role is to help users analyze fridge sales information by generating appropriate pandas code.

Key guidelines:
1. Always use the DataFrame 'df' as your data source
2. Be precise with column names and data types
3. Handle dates properly using pandas datetime functions
4. Provide clear, executable pandas code
5. Include comments explaining your analysis steps
6. Return results in a format that's easy to understand

Available columns: ID, CUSTOMER_ID, FRIDGE_MODEL, BRAND, CAPACITY_LITERS, PRICE, SALES_DATE, STORE_NAME, STORE_ADDRESS, CUSTOMER_FEEDBACK
"""
    
    def create_sources_from_df(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create sources list from DataFrame for response building."""
        sources = []
        for idx, row in df.iterrows():
            source = {
                "content": str(row.get('CUSTOMER_FEEDBACK', '')),
                "metadata": {
                    "id": str(row.get('ID', idx)),
                    "brand": str(row.get('BRAND', '')),
                    "price": str(row.get('PRICE', '')),
                    "fridge_model": str(row.get('FRIDGE_MODEL', '')),
                    "store_name": str(row.get('STORE_NAME', '')),
                    "store_address": str(row.get('STORE_ADDRESS', '')),
                    "capacity_liters": str(row.get('CAPACITY_LITERS', '')),
                    "sales_date": str(row.get('SALES_DATE', '')),
                    "feedback_rating": str(row.get('FEEDBACK_RATING', '')),
                    "score": float(row.get('PRICE', 0))
                }
            }
            sources.append(source)
        return sources

    def get_provider_config(self):
        """Delegate to provider_config module for provider settings."""
        from .provider_config import get_provider_config as _get
        return _get()
