#!/usr/bin/env python3
"""
Base profile class for configurable data processing and language support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import hashlib
import pandas as pd


@dataclass
class ColumnDefinition:
    """Definition of a CSV column."""
    name: str
    type: str = "string"
    required: bool = True
    description: str = ""
    sensitive: bool = False
    text_field: bool = False


@dataclass
class SensitizationRule:
    """Rule for data sensitization."""
    column_name: str
    prefix: str
    hash_length: int = 8
    function: Optional[Callable] = None


@dataclass
class DocumentTemplate:
    """Template for document content generation."""
    template: str
    metadata_fields: List[str] = field(default_factory=list)


class BaseProfile(ABC):
    """Abstract base class for all data processing profiles."""
    
    def __init__(self):
        self.language: str = "en"
        self.locale: str = "en_US"
        self.encoding: str = "utf-8"
        
        # Data schema
        self.required_columns: List[str] = []
        self.column_definitions: Dict[str, ColumnDefinition] = {}
        self.text_columns: List[str] = []
        self.sensitive_columns: List[str] = []
        
        # Sensitization rules
        self.sensitization_rules: Dict[str, SensitizationRule] = {}
        
        # Document generation
        self.document_template: DocumentTemplate = None
        
        # File paths
        self.data_file_path: str = ""
        self.test_data_path: str = ""
        
        # Profile name (will be set by subclasses)
        self.profile_name: str = ""
        
        # Initialize profile-specific settings
        self._initialize_profile()
    
    @abstractmethod
    def _initialize_profile(self):
        """Initialize profile-specific settings. Must be implemented by subclasses."""
        pass
    
    def get_required_columns(self) -> List[str]:
        """Get list of required columns."""
        return self.required_columns
    
    def get_column_definition(self, column_name: str) -> Optional[ColumnDefinition]:
        """Get column definition by name."""
        return self.column_definitions.get(column_name)
    
    def get_text_columns(self) -> List[str]:
        """Get list of text columns that need special processing."""
        return self.text_columns
    
    def get_sensitive_columns(self) -> List[str]:
        """Get list of columns that need sensitization."""
        return self.sensitive_columns
    
    def get_sensitization_rule(self, column_name: str) -> Optional[SensitizationRule]:
        """Get sensitization rule for a column."""
        return self.sensitization_rules.get(column_name)
    
    def get_document_template(self) -> DocumentTemplate:
        """Get document content template."""
        return self.document_template
    
    def get_data_file_path(self) -> str:
        """Get the path to the data file."""
        return self.data_file_path
    
    def get_test_data_path(self) -> str:
        """Get the path to test data."""
        return self.test_data_path
    
    def get_test_directory(self) -> str:
        """Get the path to test directory for this profile."""
        # Get the profile directory (e.g., config/profiles/default_profile)
        profile_dir = Path(__file__).parent
        return str(profile_dir / "tests")
    
    def get_test_configuration(self) -> Dict[str, Any]:
        """Get test configuration for this profile."""
        return {
            'test_directory': self.get_test_directory(),
            'test_data_path': self.get_test_data_path(),
            'profile_name': self.profile_name,
            'language': self.language,
            'locale': self.locale
        }
    
    def validate_schema(self, df_columns: List[str]) -> List[str]:
        """Validate that DataFrame has required columns. Returns list of missing columns."""
        missing = set(self.required_columns) - set(df_columns)
        return list(missing)
    
    def create_document_content(self, row_data: Dict[str, Any]) -> str:
        """Create document content from row data using the template."""
        if not self.document_template:
            raise ValueError("Document template not defined")
        
        return self.document_template.template.format(**row_data)
    
    def create_document_metadata(self, row_data: Dict[str, Any], row_index: int) -> Dict[str, Any]:
        """Create document metadata from row data."""
        metadata = {"linha": row_index}
        
        for field in self.document_template.metadata_fields:
            if field in row_data:
                metadata[field] = row_data[field]
        
        return metadata
    
    def sensitize_value(self, value: Any, column_name: str, sensitive_mapping: Dict[str, str]) -> str:
        """Sensitize a value using the column's sensitization rule."""
        if pd.isna(value) or str(value).strip() == "":
            return ""
        
        rule = self.get_sensitization_rule(column_name)
        if not rule:
            return str(value)
        
        value_str = str(value).strip()
        if len(value_str) < 3:
            return value_str
        
        # Use custom function if provided, otherwise use default hashing
        if rule.function:
            return rule.function(value_str, rule, sensitive_mapping)
        else:
            return self._default_sensitize(value_str, rule, sensitive_mapping)
    
    def _default_sensitize(self, value: str, rule: SensitizationRule, sensitive_mapping: Dict[str, str]) -> str:
        """Default sensitization using MD5 hash."""
        hashed = hashlib.md5(value.encode()).hexdigest()[:rule.hash_length].upper()
        placeholder = f"{rule.prefix}_{hashed}"
        sensitive_mapping[placeholder] = value
        return placeholder
    
    def desensitize_text(self, text: str, sensitive_mapping: Dict[str, str]) -> str:
        """Desensitize text by replacing placeholders with original values."""
        if not isinstance(text, str):
            return str(text)
        
        result = text
        for placeholder, original in sensitive_mapping.items():
            result = result.replace(placeholder, original)
        return result
    
    def get_prompt_template(self) -> str:
        """Get the prompt template for the RAG system."""
        return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """Default prompt template."""
        return """
        You are a customer service data analysis assistant.
        Your role is to answer questions based on the provided data.

        Context information:
        {context}

        Question: {question}

        Instructions:
        1. Use only the provided context information to answer the question.
        2. Be specific and cite relevant data when appropriate.
        3. If there is not enough information to answer, state that clearly.
        4. Always answer in {language}.
        5. When mentioning scores, dealers, or other specific data, be precise.

        Answer:
        """.format(language=self.language)
    
    def get_report_config(self) -> Dict[str, Any]:
        """Get report generation configuration."""
        return {
            "title": f"Data Report ({self.language})",
            "max_rows": 500,
            "date_column": "CREATE_DATE",
            "score_column": "SCORE",
            "dealer_column": "DEALER_CODE",
            "repair_type_column": "REPAIR_TYPE_NAME"
        }
