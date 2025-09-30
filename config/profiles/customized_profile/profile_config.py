#!/usr/bin/env python3
"""
Brazilian Portuguese profile for NPS data processing.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from ..base_profile import BaseProfile, ColumnDefinition, SensitizationRule, DocumentTemplate
from core.rag.generic_data_processor import DataSchema
from reports.generic_report_builder import ReportConfig


class CustomizedProfile(BaseProfile):
    """Customized profile for NPS data processing."""
    
    def _initialize_profile(self):
        """Initialize Brazilian Portuguese profile settings."""
        # Profile identification
        self.profile_name = "customized_profile"
        
        # Language settings
        self.language = "pt-BR"
        self.locale = "pt_BR"
        
        # File paths
        self.data_file_path = str(Path(__file__).parent / "test_data" / "csv.csv")
        self.test_data_path = str(Path(__file__).parent / "test_data")
        self.test_directory = str(Path(__file__).parent / "tests")
        
        # Define required columns (extracted from original hardcoded list)
        self.required_columns = [
            'RO_NO', 'DEALER_CODE', 'SUB_DEALER_CODE', 'SCORE', 'SERVICE_ATTITUDE',
            'ENVIRONMENT', 'EFFICIENCY', 'EFFECTIVENESS', 'PARTS_AVAILABILITY',
            'OTHERS', 'TROUBLE_DESC', 'CHECK_RESULT', 'REPAIR_TYPE_NAME', 'VIN',
            'CREATE_DATE', 'OTHERS_REASON'
        ]
        
        # Define column definitions
        self.column_definitions = {
            'RO_NO': ColumnDefinition('RO_NO', 'int', True, "Repair Order Number"),
            'DEALER_CODE': ColumnDefinition('DEALER_CODE', 'string', True, "Dealer Code", sensitive=True),
            'SUB_DEALER_CODE': ColumnDefinition('SUB_DEALER_CODE', 'string', True, "Sub Dealer Code", sensitive=True),
            'SCORE': ColumnDefinition('SCORE', 'float', True, "NPS Score"),
            'SERVICE_ATTITUDE': ColumnDefinition('SERVICE_ATTITUDE', 'string', True, "Service Attitude Rating"),
            'ENVIRONMENT': ColumnDefinition('ENVIRONMENT', 'string', True, "Environment Rating"),
            'EFFICIENCY': ColumnDefinition('EFFICIENCY', 'string', True, "Efficiency Rating"),
            'EFFECTIVENESS': ColumnDefinition('EFFECTIVENESS', 'string', True, "Effectiveness Rating"),
            'PARTS_AVAILABILITY': ColumnDefinition('PARTS_AVAILABILITY', 'string', True, "Parts Availability Rating"),
            'OTHERS': ColumnDefinition('OTHERS', 'string', True, "Other Rating"),
            'TROUBLE_DESC': ColumnDefinition('TROUBLE_DESC', 'string', True, "Trouble Description", text_field=True),
            'CHECK_RESULT': ColumnDefinition('CHECK_RESULT', 'string', True, "Check Result", text_field=True),
            'REPAIR_TYPE_NAME': ColumnDefinition('REPAIR_TYPE_NAME', 'string', True, "Repair Type Name"),
            'VIN': ColumnDefinition('VIN', 'string', True, "Vehicle Identification Number", sensitive=True),
            'CREATE_DATE': ColumnDefinition('CREATE_DATE', 'datetime', True, "Creation Date"),
            'OTHERS_REASON': ColumnDefinition('OTHERS_REASON', 'string', True, "Other Reason", text_field=True)
        }
        
        # Define text columns (extracted from original hardcoded logic)
        self.text_columns = ['TROUBLE_DESC', 'CHECK_RESULT', 'OTHERS_REASON']
        
        # Define sensitive columns (extracted from original hardcoded logic)
        self.sensitive_columns = ['VIN', 'DEALER_CODE', 'SUB_DEALER_CODE']
        
        # Define sensitization rules (extracted from original hardcoded logic)
        self.sensitization_rules = {
            'VIN': SensitizationRule('VIN', 'VIN', 8),
            'DEALER_CODE': SensitizationRule('DEALER_CODE', 'DEALER', 6),
            'SUB_DEALER_CODE': SensitizationRule('SUB_DEALER_CODE', 'SUB_DEALER', 6)
        }
        
        # Define document template (extracted from original hardcoded Portuguese text)
        self.document_template = DocumentTemplate(
            template=(
                "Ordem {RO_NO} do dealer {DEALER_CODE} "
                "(sub-dealer {SUB_DEALER_CODE}) recebeu uma pontuação geral de {SCORE}. "
                "Atendimento: {SERVICE_ATTITUDE}, Ambiente: {ENVIRONMENT}, "
                "Eficiência: {EFFICIENCY}, Efetividade: {EFFECTIVENESS}, "
                "Peças: {PARTS_AVAILABILITY}, Outros: {OTHERS}. "
                "Descrição do problema: {TROUBLE_DESC}. "
                "Resultado da verificação: {CHECK_RESULT}. "
                "Tipo de reparo: {REPAIR_TYPE_NAME}. "
                "VIN: {VIN}. "
                "Data da ordem: {CREATE_DATE}. "
                "Motivo adicional: {OTHERS_REASON}."
            ),
            metadata_fields=[
                "ro_no", "vin", "dealer", "data_ordem", "score", "repair_type"
            ]
        )
    
    def get_prompt_template(self) -> str:
        """Get Portuguese prompt template for the RAG system."""
        return """
        Você é um assistente de análise de dados de atendimento ao cliente.
        Seu papel é responder perguntas com base nos dados de NPS (Net Promoter Score) fornecidos.

        Informações de contexto:
        {context}

        Pergunta: {question}

        Instruções:
        1. Use apenas as informações de contexto fornecidas para responder à pergunta.
        2. Seja específico e cite dados relevantes quando apropriado.
        3. Se não houver informações suficientes para responder, declare isso claramente.
        4. Sempre responda em português brasileiro.
        5. Ao mencionar pontuações, dealers ou outros dados específicos, seja preciso.

        Resposta:
        """
    
    def get_report_config(self) -> Dict[str, Any]:
        """Get Portuguese report generation configuration."""
        return {
            "title": "Relatório de Dados NPS",
            "max_rows": 500,
            "date_column": "CREATE_DATE",
            "score_column": "SCORE",
            "dealer_column": "DEALER_CODE",
            "repair_type_column": "REPAIR_TYPE_NAME",
            "language": "pt-BR"
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize data according to BR profile rules."""
        # Clean text columns
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Clean score column
        if 'SCORE' in df.columns:
            score_num = pd.to_numeric(
                df['SCORE'].astype(str).str.replace(',', '.', regex=False), 
                errors='coerce'
            )
            df['SCORE'] = score_num.fillna(0).astype(float)
        
        # Clean date column
        if 'CREATE_DATE' in df.columns:
            df['CREATE_DATE'] = pd.to_datetime(df['CREATE_DATE'], errors='coerce')
        
        return df
    
    def create_document_metadata(self, row_data: Dict[str, Any], row_index: int) -> Dict[str, Any]:
        """Create document metadata with Portuguese field names."""
        metadata = {"linha": row_index}
        
        # Map to Portuguese metadata field names
        field_mapping = {
            "ro_no": "RO_NO",
            "vin": "VIN", 
            "dealer": "DEALER_CODE",
            "data_ordem": "CREATE_DATE",
            "score": "SCORE",
            "repair_type": "REPAIR_TYPE_NAME"
        }
        
        for metadata_field, csv_field in field_mapping.items():
            if csv_field in row_data:
                metadata[metadata_field] = row_data[csv_field]
        
        return metadata
    
    def get_test_directory(self) -> str:
        """Get the path to test directory for BR profile."""
        return self.test_directory
    
    def get_data_schema(self) -> DataSchema:
        """Get the data schema for BR profile."""
        return DataSchema(
            required_columns=self.required_columns,
            sensitive_columns=['DEALER_CODE', 'SUB_DEALER_CODE', 'VIN'],
            date_columns=['CREATE_DATE', 'SUBMIT_DATE', 'UPDATE_DATE'],
            text_columns=['TROUBLE_DESC', 'CHECK_RESULT', 'OTHERS_REASON'],
            metadata_columns=['RO_NO', 'SCORE', 'SERVICE_ATTITUDE', 'ENVIRONMENT', 'EFFICIENCY', 'EFFECTIVENESS', 'PARTS_AVAILABILITY', 'OTHERS', 'REPAIR_TYPE_NAME'],
            id_column='RO_NO',
            score_column='SCORE'
        )
    
    def get_report_config(self) -> ReportConfig:
        """Get the report configuration for BR profile."""
        return ReportConfig(
            title="NPS Report - Brazilian Portuguese",
            date_columns=['CREATE_DATE'],
            score_columns=['SCORE'],
            filter_columns=['DEALER_CODE', 'REPAIR_TYPE_NAME'],
            display_columns=['RO_NO', 'DEALER_CODE', 'SCORE', 'SERVICE_ATTITUDE', 'ENVIRONMENT', 'EFFICIENCY', 'EFFECTIVENESS', 'PARTS_AVAILABILITY', 'OTHERS', 'TROUBLE_DESC', 'REPAIR_TYPE_NAME', 'CREATE_DATE'],
            max_rows=500
        )

    def get_censoring_mappings(self) -> Dict[str, str]:
        """Get censoring mappings for sensitive data."""
        return {}
    
    def get_schema_hints(self, sample_data: str) -> str:
        """Get schema hints for the data."""
        return f"""
Schema hints for NPS data:
- RO_NO: Repair Order Number (unique identifier)
- DEALER_CODE: Dealer code (sensitive)
- SUB_DEALER_CODE: Sub-dealer code (sensitive) 
- SCORE: NPS Score (0-10 rating)
- SERVICE_ATTITUDE: Service attitude rating
- ENVIRONMENT: Environment rating
- EFFICIENCY: Efficiency rating
- EFFECTIVENESS: Effectiveness rating
- PARTS_AVAILABILITY: Parts availability rating
- OTHERS: Other rating
- TROUBLE_DESC: Trouble description (text field)
- CHECK_RESULT: Check result (text field)
- REPAIR_TYPE_NAME: Type of repair performed
- VIN: Vehicle identification number (sensitive)
- CREATE_DATE: Date when order was created
- OTHERS_REASON: Additional reason (text field)
"""
    
    def get_llm_system_prompt(self) -> str:
        """Get system prompt for LLM interactions."""
        return """
You are a data analysis assistant for NPS (Net Promoter Score) data.
Your role is to help users analyze customer satisfaction information by generating appropriate pandas code.

Key guidelines:
1. Always use the DataFrame 'df' as your data source
2. Be precise with column names and data types
3. Handle dates properly using pandas datetime functions
4. Provide clear, executable pandas code
5. Include comments explaining your analysis steps
6. Return results in a format that's easy to understand

Available columns: RO_NO, DEALER_CODE, SUB_DEALER_CODE, SCORE, SERVICE_ATTITUDE, ENVIRONMENT, EFFICIENCY, EFFECTIVENESS, PARTS_AVAILABILITY, OTHERS, TROUBLE_DESC, CHECK_RESULT, REPAIR_TYPE_NAME, VIN, CREATE_DATE, OTHERS_REASON
"""
    
    def create_sources_from_df(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create sources list from DataFrame for response building."""
        sources = []
        for idx, row in df.iterrows():
            source = {
                "content": str(row.get('TROUBLE_DESC', '') + ' ' + str(row.get('CHECK_RESULT', '')) + ' ' + str(row.get('OTHERS_REASON', ''))),
                "metadata": {
                    "ro_no": str(row.get('RO_NO', idx)),
                    "dealer_code": str(row.get('DEALER_CODE', '')),
                    "score": float(row.get('SCORE', 0)),
                    "service_attitude": str(row.get('SERVICE_ATTITUDE', '')),
                    "environment": str(row.get('ENVIRONMENT', '')),
                    "efficiency": str(row.get('EFFICIENCY', '')),
                    "effectiveness": str(row.get('EFFECTIVENESS', '')),
                    "parts_availability": str(row.get('PARTS_AVAILABILITY', '')),
                    "others": str(row.get('OTHERS', '')),
                    "repair_type": str(row.get('REPAIR_TYPE_NAME', '')),
                    "vin": str(row.get('VIN', '')),
                    "create_date": str(row.get('CREATE_DATE', '')),
                    "score": float(row.get('SCORE', 0))
                }
            }
            sources.append(source)
        return sources

    def get_provider_config(self):
        """Delegate to provider_config module for provider settings."""
        from .provider_config import get_provider_config as _get
        return _get()
