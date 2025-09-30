#!/usr/bin/env python3
"""
Setup script for NPS test data.
Ensures all required test data files are available for testing the customized_profile.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def create_nps_test_dataframe():
    """Create a comprehensive NPS test DataFrame."""
    return pd.DataFrame({
        'RO_NO': ['C0125030811193253828', 'C0125021911003152009', 'C0125031009471654086', 'C0125031500035355209', 'C0125030814324153842'],
        'DEALER_CODE': ['BYDAMEBR0007W', 'BYDAMEBR0005W', 'BYDAMEBR0007W', 'BYDAMEBR0007W', 'BYDAMEBR0007W'],
        'SUB_DEALER_CODE': ['', '', '', '', ''],
        'SCORE': [9.0, 10.0, 9.0, 8.0, 7.0],
        'SERVICE_ATTITUDE': ['Y', '', 'Y', 'Y', 'Y'],
        'ENVIRONMENT': ['', 'Y', '', '', ''],
        'EFFICIENCY': ['', 'Y', 'Y', '', ''],
        'EFFECTIVENESS': ['', 'Y', '', '', ''],
        'PARTS_AVAILABILITY': ['', '', '', '', ''],
        'OTHERS': ['', '', '', 'Y', ''],
        'TROUBLE_DESC': ['', 'CLIENTE SOLICITA REVISAO DE 20.000 KM R$360,00', '', 'Fornecer manual do carro físico byd king E enviar nota fiscal das revisões do carro', 'Demorou bastante e peguei o carro sem ter terminado, aguardando chegar a peça para finalizar o conserto'],
        'CHECK_RESULT': ['', '', '', '', ''],
        'REPAIR_TYPE_NAME': ['Other Repair', 'Other Repair', 'Other Repair', 'Other Repair', 'Other Repair'],
        'VIN': ['LC0CE4CC2R0009877', 'LGXCE4CC7S0004466', 'LGXC74C43R0011170', 'LC0C76C44S0027303', 'LGXC74C4XS0000513'],
        'CREATE_DATE': ['2025-03-09 00:01:22.000', '2025-02-20 00:03:10.000', '2025-03-11 00:03:24.000', '2025-03-16 00:02:14.000', '2025-03-09 00:01:22.000'],
        'OTHERS_REASON': ['', '', '', 'Y', '']
    })

def create_extended_nps_test_dataframe():
    """Create an extended NPS test DataFrame with more data."""
    base_data = create_nps_test_dataframe()
    
    # Add more rows with different dealers and scenarios
    additional_data = pd.DataFrame({
        'RO_NO': ['C0125032413255855782', 'C0125031019470354119', 'C0125030815383052978', 'C0125030808101153814', 'C0125030811193253829'],
        'DEALER_CODE': ['BYDAMEBR0007W', 'BYDAMEBR0007W', 'BYDAMEBR0045W', 'BYDAMEBR0045W', 'BYDAMEBR0005W'],
        'SUB_DEALER_CODE': ['', '', '', '', ''],
        'SCORE': [9.0, 10.0, 10.0, 10.0, 8.0],
        'SERVICE_ATTITUDE': ['Y', 'Y', 'Y', 'Y', 'Y'],
        'ENVIRONMENT': ['', '', 'Y', 'Y', ''],
        'EFFICIENCY': ['', 'Y', 'Y', 'Y', ''],
        'EFFECTIVENESS': ['', '', 'Y', 'Y', ''],
        'PARTS_AVAILABILITY': ['', '', '', '', 'Y'],
        'OTHERS': ['', '', '', 'Y', ''],
        'TROUBLE_DESC': ['', '', '', '', 'Cliente solicitou revisão completa do veículo'],
        'CHECK_RESULT': ['', '', '', '', 'Revisão realizada com sucesso'],
        'REPAIR_TYPE_NAME': ['Other Repair', 'Other Repair', 'Other Repair', 'Other Repair', 'Maintenance'],
        'VIN': ['LGXC74C40R0014141', 'LC0CE4CC9R0008497', 'LGXC74C44R0009167', 'LGXCE4CC9S0010057', 'LGXC74C40R0014142'],
        'CREATE_DATE': ['2025-03-25 00:03:33.000', '2025-03-11 00:03:24.000', '2025-03-09 00:01:22.000', '2025-03-09 00:01:22.000', '2025-03-09 00:01:22.000'],
        'OTHERS_REASON': ['', '', '', '', 'Revisão preventiva solicitada pelo cliente']
    })
    
    return pd.concat([base_data, additional_data], ignore_index=True)

def create_nps_test_queries():
    """Create NPS-specific test queries for different scenarios."""
    return {
        "score_analysis_queries": [
            "What is the average NPS score across all dealers?",
            "Which dealer has the highest average NPS score?",
            "How many repairs received a score of 10?",
            "What percentage of repairs have scores above 8?"
        ],
        "dealer_performance_queries": [
            "Compare NPS performance between BYDAMEBR0007W and BYDAMEBR0005W",
            "Which dealer has the most repairs?",
            "Show me all repairs for dealer BYDAMEBR0045W",
            "What is the average score for BYDAMEBR0007W?"
        ],
        "service_quality_queries": [
            "How many repairs had positive service attitude ratings?",
            "Which repair types have the best efficiency ratings?",
            "Compare environment ratings across different dealers",
            "What percentage of repairs have all service quality indicators positive?"
        ],
        "complex_nps_queries": [
            "Analyze the relationship between repair type and NPS score",
            "Which dealers have the best combination of high scores and positive service ratings?",
            "What are the main issues mentioned in trouble descriptions?",
            "Compare NPS trends over time for different dealers"
        ]
    }

def create_nps_test_responses():
    """Create mock NPS test responses."""
    return {
        "text2query_success": {
            "answer": "The average NPS score across all dealers is 9.1",
            "sources": [{"dealer": "BYDAMEBR0007W", "score": 9.0}],
            "confidence": "high",
            "query_type": "aggregation",
            "synthesis_method": "traditional"
        },
        "text2query_failure": {
            "error": "No results found for the specified criteria",
            "query_type": "synthesis_error"
        },
        "rag_success": {
            "answer": "Based on the NPS data, BYDAMEBR0045W has the highest average score of 10.0 with excellent service attitude and environment ratings.",
            "sources": [{"content": "NPS repair data", "metadata": {"dealer": "BYDAMEBR0045W", "score": 10.0}}],
            "confidence": "medium"
        },
        "rag_failure": {
            "answer": "Não foi possível encontrar informações específicas sobre o desempenho dos dealers.",
            "sources": [],
            "confidence": "low"
        }
    }

def setup_test_data():
    """Setup all NPS test data files."""
    print("Setting up NPS test data for customized_profile...")
    
    # Create test data directory
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Create basic NPS test DataFrame
    df_basic = create_nps_test_dataframe()
    df_basic.to_csv(test_data_dir / "nps_basic.csv", index=False)
    print(f"✅ Created basic NPS test data: {len(df_basic)} rows")
    
    # Create extended NPS test DataFrame
    df_extended = create_extended_nps_test_dataframe()
    df_extended.to_csv(test_data_dir / "nps_extended.csv", index=False)
    print(f"✅ Created extended NPS test data: {len(df_extended)} rows")
    
    # Create NPS test queries
    test_queries = create_nps_test_queries()
    with open(test_data_dir / "nps_test_queries.json", "w") as f:
        json.dump(test_queries, f, indent=2)
    print(f"✅ Created NPS test queries: {sum(len(queries) for queries in test_queries.values())} queries")
    
    # Create NPS test responses
    test_responses = create_nps_test_responses()
    with open(test_data_dir / "nps_test_responses.json", "w") as f:
        json.dump(test_responses, f, indent=2)
    print(f"✅ Created NPS test responses: {len(test_responses)} response types")
    
    # Create NPS test configuration
    test_config = {
        "test_data_files": {
            "basic": "nps_basic.csv",
            "extended": "nps_extended.csv",
            "original": "csv.csv"
        },
        "test_queries_file": "nps_test_queries.json",
        "test_responses_file": "nps_test_responses.json",
        "data_schema": {
            "required_columns": ["RO_NO", "DEALER_CODE", "SCORE", "SERVICE_ATTITUDE", "REPAIR_TYPE_NAME", "VIN", "CREATE_DATE"],
            "sensitive_columns": ["VIN", "DEALER_CODE", "SUB_DEALER_CODE"],
            "date_columns": ["CREATE_DATE", "SUBMIT_DATE", "UPDATE_DATE"],
            "text_columns": ["TROUBLE_DESC", "CHECK_RESULT", "OTHERS_REASON"],
            "numeric_columns": ["SCORE"],
            "rating_columns": ["SERVICE_ATTITUDE", "ENVIRONMENT", "EFFICIENCY", "EFFECTIVENESS", "PARTS_AVAILABILITY", "OTHERS"]
        },
        "language": "pt-BR",
        "profile_name": "customized_profile"
    }
    
    with open(test_data_dir / "nps_test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)
    print("✅ Created NPS test configuration")
    
    print()
    print("NPS test data setup complete!")
    print(f"Test data directory: {test_data_dir}")
    print()
    print("Available test files:")
    for file_path in test_data_dir.glob("*"):
        print(f"  - {file_path.name}")

def verify_test_data():
    """Verify that all NPS test data files exist and are valid."""
    print("Verifying NPS test data...")
    
    test_data_dir = Path(__file__).parent / "test_data"
    
    required_files = [
        "nps_basic.csv",
        "nps_extended.csv",
        "nps_test_queries.json",
        "nps_test_responses.json",
        "nps_test_config.json",
        "csv.csv"  # Original data file
    ]
    
    all_valid = True
    
    for file_name in required_files:
        file_path = test_data_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} - Found")
            
            # Validate CSV files
            if file_name.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    print(f"   - {len(df)} rows, {len(df.columns)} columns")
                    
                    # Check for required columns in NPS data
                    if file_name.startswith('nps_'):
                        required_nps_columns = ['RO_NO', 'DEALER_CODE', 'SCORE', 'CREATE_DATE']
                        missing_columns = set(required_nps_columns) - set(df.columns)
                        if missing_columns:
                            print(f"   ⚠️  Missing NPS columns: {missing_columns}")
                        else:
                            print(f"   ✅ All required NPS columns present")
                            
                except Exception as e:
                    print(f"   ❌ Error reading CSV: {e}")
                    all_valid = False
            
            # Validate JSON files
            elif file_name.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    print(f"   - Valid JSON")
                except Exception as e:
                    print(f"   ❌ Error reading JSON: {e}")
                    all_valid = False
        else:
            print(f"❌ {file_name} - Missing")
            all_valid = False
    
    if all_valid:
        print()
        print("✅ All NPS test data files are valid!")
    else:
        print()
        print("❌ Some NPS test data files are invalid or missing!")
    
    return all_valid

def main():
    """Main setup function."""
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        return 0 if verify_test_data() else 1
    else:
        setup_test_data()
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
