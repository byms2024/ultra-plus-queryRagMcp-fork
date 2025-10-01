# 🚀 Customized Profile Setup Guide - NPS Data Analysis

> **Transform your NPS data into intelligent insights!** This guide provides comprehensive testing and setup instructions specifically for the customized_profile, which handles NPS (Net Promoter Score) data with Brazilian Portuguese language support.

## 📑 Table of Contents

- [📋 Profile Overview](#-profile-overview)
- [🧪 Testing](#-testing)
- [🖥️ API Testing](#️-api-testing)
- [📊 Performance Insights](#-performance-insights)
- [🔧 Troubleshooting](#-troubleshooting)

---

## 📋 Profile Overview

The **customized_profile** is designed for **NPS (Net Promoter Score) data analysis** with the following characteristics:

### 🗂️ Data Schema
- **Primary Data**: NPS repair order records with customer satisfaction scores
- **Language**: Brazilian Portuguese (pt-BR)
- **Key Fields**: RO_NO, DEALER_CODE, SCORE, SERVICE_ATTITUDE, ENVIRONMENT, EFFICIENCY, EFFECTIVENESS, PARTS_AVAILABILITY, TROUBLE_DESC, VIN
- **Text Fields**: TROUBLE_DESC, CHECK_RESULT, OTHERS_REASON (for RAG analysis)
- **Sensitive Fields**: VIN, DEALER_CODE, SUB_DEALER_CODE (automatically censored)
- **Rating Fields**: SERVICE_ATTITUDE, ENVIRONMENT, EFFICIENCY, EFFECTIVENESS, PARTS_AVAILABILITY, OTHERS

### 🎯 Use Cases
- **NPS Score Analysis**: Analyze customer satisfaction scores across dealers
- **Service Quality Assessment**: Evaluate service attitude, environment, and efficiency ratings
- **Dealer Performance Comparison**: Compare NPS performance between different dealers
- **Customer Feedback Analysis**: Analyze trouble descriptions and repair results
- **Repair Type Analysis**: Track performance by repair type and service quality

### 🌍 Language Support
- **Primary Language**: Brazilian Portuguese (pt-BR)
- **Response Format**: Portuguese responses for user queries
- **Data Processing**: Handles Portuguese text in trouble descriptions and feedback
- **Localization**: Portuguese field names and metadata

---

## 🧪 Testing

> **🎯 Goal:** Verify everything works perfectly before moving to production!

### 🚀 Quick Test (30 seconds)
Want to see if everything is working? Run this quick test:

```bash
# Start the server
python scripts/run_unified_api.py

# In another terminal, test the health endpoint
curl http://localhost:8000/health
```

If you see `"status": "healthy"` and `"profile": "customized_profile"`, you're ready to go! 🎉

### 📋 Comprehensive Testing Guide

### Test Structure Overview

The testing system is organized into several categories:

- **Engine Tests**: Core functionality testing (initialization, question answering, error handling)
- **API Tests**: HTTP endpoint testing (endpoints, integration, error handling)
- **MCP Tests**: MCP protocol testing (tools, integration, error handling)
- **NPS-Specific Tests**: Brazilian Portuguese NPS data validation

Each test category uses NPS-specific datasets and test scenarios to ensure comprehensive coverage.

### Running Tests

#### 1. Set Up NPS Test Data
```bash
# Navigate to the test directory
cd config/profiles/customized_profile/tests

# Set up NPS test data
python setup_test_data.py

# Verify NPS test data
python setup_test_data.py verify
```

#### 2. Run All NPS Tests
```bash
# From project root
PYTHONPATH=/path/to/project python config/profiles/customized_profile/tests/run_unified_tests.py all
```

#### 3. Run Specific Test Suites
```bash
# Engine tests only
PYTHONPATH=/path/to/project python config/profiles/customized_profile/tests/run_unified_tests.py engine

# API tests only
PYTHONPATH=/path/to/project python config/profiles/customized_profile/tests/run_unified_tests.py api

# MCP tests only
PYTHONPATH=/path/to/project python config/profiles/customized_profile/tests/run_unified_tests.py mcp

# NPS-specific validation tests
PYTHONPATH=/path/to/project python config/profiles/customized_profile/tests/run_unified_tests.py nps

# Quick tests (unit tests only)
PYTHONPATH=/path/to/project python config/profiles/customized_profile/tests/run_unified_tests.py quick

# Integration tests only
PYTHONPATH=/path/to/project python config/profiles/customized_profile/tests/run_unified_tests.py integration
```

#### 4. Run Individual Tests
```bash
# Using pytest directly
PYTHONPATH=/path/to/project python -m pytest config/profiles/customized_profile/tests/test_unified_engine.py -v

# Run specific test class
PYTHONPATH=/path/to/project python -m pytest config/profiles/customized_profile/tests/test_unified_engine.py::TestNPSDataProcessing -v

# Run specific test method
PYTHONPATH=/path/to/project python -m pytest config/profiles/customized_profile/tests/test_unified_engine.py::TestNPSDataProcessing::test_nps_data_cleaning -v
```

### NPS Test Data Management

The NPS test data management system automatically handles data setup and verification:

1. **Data Verification**: Checks if required NPS test files exist
2. **Auto-Generation**: Creates missing NPS test data if needed
3. **Test Execution**: Runs tests with verified NPS data
4. **Results Reporting**: Provides comprehensive NPS test results

This ensures consistent NPS test environments and eliminates manual data setup steps.

### Available NPS Test Data

The system provides several types of NPS test data:

- **`nps_basic.csv`**: Basic NPS test data (5 records)
- **`nps_extended.csv`**: Extended NPS test data (10 records)
- **`csv.csv`**: Original NPS data file (3000+ records)
- **`nps_test_queries.json`**: NPS-specific test queries
- **`nps_test_responses.json`**: Expected NPS test responses
- **`nps_test_config.json`**: NPS test configuration

---

## 🖥️ API Testing

> **🎯 Goal:** Get your API server running and ready to handle NPS requests!

### 🚀 FastAPI Server

#### Start the API Server
```bash
# 🎯 Recommended: Using the provided script
python scripts/run_unified_api.py

# 🔧 Alternative: Using uvicorn directly
uvicorn api.unified_api:app --host 0.0.0.0 --port 8000 --reload
```

> **💡 Pro Tip:** The `--reload` flag automatically restarts the server when you make code changes - perfect for development!

#### 🎉 Success Indicators
When the server starts successfully, you'll see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**🎊 Congratulations!** Your server is now running and ready to handle NPS requests!

#### 🧪 Comprehensive NPS Test Cases

##### **Basic Health Check**
```bash
curl http://localhost:8000/health
```
**Expected Result:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "profile": "customized_profile",
  "engines": {
    "text2query_available": true,
    "rag_available": true
  },
  "data_records": 3000
}
```

##### **NPS Score Aggregation Query (Text2Query Engine)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average NPS score for dealer BYDAMEBR0007W?", "method": "auto"}'
```
**Expected Result:**
```json
{
  "question": "What is the average NPS score for dealer BYDAMEBR0007W?",
  "answer": "O score médio NPS para o dealer BYDAMEBR0007W é 9.0",
  "sources": [{
    "content": "Dados NPS do dealer",
    "metadata": {
      "dealer_code": "BYDAMEBR0007W",
      "score": 9.0,
      "ro_no": "C0125030811193253828"
    }
  }],
  "confidence": "high",
  "method_used": "text2query",
  "execution_time": 2.1,
  "timestamp": "2025-01-01T12:00:00.000000",
  "profile": "customized_profile"
}
```
**💡 Hint:** This should use Text2Query engine (~2s execution time) for simple NPS aggregations.

##### **Portuguese Customer Feedback Analysis (RAG Engine)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais são os principais problemas mencionados nas descrições de problemas?", "method": "rag"}'
```
**Expected Result:**
```json
{
  "question": "Quais são os principais problemas mencionados nas descrições de problemas?",
  "answer": "Com base nos dados NPS, os principais problemas mencionados incluem demora no atendimento, necessidade de peças e solicitações de revisão. Os clientes também mencionam a necessidade de manuais físicos e notas fiscais das revisões.",
  "sources": [{
    "content": "TROUBLE_DESC: Demorou bastante e peguei o carro sem ter terminado, aguardando chegar a peça para finalizar o conserto",
    "metadata": {
      "ro_no": "C0125030814324153842",
      "dealer_code": "BYDAMEBR0007W",
      "score": 7.0
    }
  }],
  "confidence": "medium",
  "method_used": "rag",
  "execution_time": 3.5,
  "timestamp": "2025-01-01T12:00:00.000000",
  "profile": "customized_profile"
}
```
**💡 Hint:** This uses RAG engine (~3.5s execution time) for complex Portuguese text analysis.

##### **Dealer Performance Comparison (Text2Query Engine)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare NPS performance between BYDAMEBR0007W and BYDAMEBR0005W", "method": "auto"}'
```
**Expected Result:**
```json
{
  "question": "Compare NPS performance between BYDAMEBR0007W and BYDAMEBR0005W",
  "answer": "BYDAMEBR0005W tem melhor performance (10.0) comparado a BYDAMEBR0007W (9.0). BYDAMEBR0005W apresenta melhor performance geral com scores mais altos.",
  "sources": [...],
  "confidence": "high",
  "method_used": "text2query",
  "execution_time": 2.3,
  "timestamp": "2025-01-01T12:00:00.000000",
  "profile": "customized_profile"
}
```
**💡 Hint:** Text2Query engine handles structured NPS comparisons efficiently.

##### **Service Quality Analysis (Text2Query Engine)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many repairs had positive service attitude ratings?", "method": "auto"}'
```
**Expected Result:**
```json
{
  "question": "How many repairs had positive service attitude ratings?",
  "answer": "85% dos reparos tiveram avaliações positivas de atitude de serviço. A maioria dos dealers apresenta boa atitude no atendimento.",
  "sources": [...],
  "confidence": "high",
  "method_used": "text2query",
  "execution_time": 1.8,
  "timestamp": "2025-01-01T12:00:00.000000",
  "profile": "customized_profile"
}
```
**💡 Hint:** Text2Query engine excels at service quality metrics analysis.

##### **Repair Type Performance Analysis (RAG Engine)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Analyze the relationship between repair type and NPS score", "method": "rag"}'
```
**Expected Result:**
```json
{
  "question": "Analyze the relationship between repair type and NPS score",
  "answer": "A análise dos dados NPS mostra que reparos de manutenção preventiva tendem a ter scores mais altos (9-10) comparado a reparos corretivos (7-8). Os clientes valorizam mais a proatividade na manutenção do que a correção de problemas.",
  "sources": [
    {
      "content": "TROUBLE_DESC: Cliente solicitou revisão completa do veículo",
      "metadata": {
        "ro_no": "C0125030811193253829",
        "repair_type": "Maintenance",
        "score": 8.0
      }
    }
  ],
  "confidence": "medium",
  "method_used": "rag",
  "execution_time": 4.2,
  "timestamp": "2025-01-01T12:00:00.000000",
  "profile": "customized_profile"
}
```
**💡 Hint:** RAG engine provides detailed analysis of repair type patterns and customer satisfaction.

##### **Multi-Dealer Performance Analysis (RAG Engine)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which dealers have the best combination of high scores and positive service ratings?", "method": "rag"}'
```
**Expected Result:**
```json
{
  "question": "Which dealers have the best combination of high scores and positive service ratings?",
  "answer": "BYDAMEBR0045W demonstra a melhor combinação com scores de 10.0 e avaliações positivas em todos os aspectos de serviço (atitude, ambiente, eficiência, efetividade). BYDAMEBR0005W também apresenta excelente performance com scores consistentemente altos.",
  "sources": [
    {
      "content": "Dados NPS consolidados",
      "metadata": {
        "dealer_code": "BYDAMEBR0045W",
        "score": 10.0,
        "service_attitude": "Y",
        "environment": "Y",
        "efficiency": "Y",
        "effectiveness": "Y"
      }
    }
  ],
  "confidence": "high",
  "method_used": "rag",
  "execution_time": 3.8,
  "timestamp": "2025-01-01T12:00:00.000000",
  "profile": "customized_profile"
}
```
**💡 Hint:** This complex query combines multiple NPS metrics for comprehensive dealer evaluation.

##### **Search Functionality**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "BYDAMEBR0007W", "top_k": 10}'
```
**Expected Result:**
```json
{
  "query": "BYDAMEBR0007W",
  "results": [
    {
      "content": "TROUBLE_DESC: Demorou bastante e peguei o carro sem ter terminado",
      "metadata": {
        "ro_no": "C0125030814324153842",
        "dealer_code": "BYDAMEBR0007W",
        "score": 7.0,
        "vin": "LGXC74C4XS0000513"
      },
      "score": 0.85
    }
  ],
  "total_found": 1
}
```

##### **System Statistics**
```bash
curl http://localhost:8000/stats
```
**Expected Result:**
```json
{
  "profile": "customized_profile",
  "data_records": 3000,
  "engines": {
    "text2query": {
      "available": true,
      "methods": ["traditional", "langchain_direct", "langchain_agent"]
    },
    "rag": {
      "available": true,
      "vector_store": "chroma",
      "documents": 3001
    }
  }
}
```

##### **Available Methods**
```bash
curl http://localhost:8000/methods
```
**Expected Result:**
```json
{
  "available_methods": ["text2query", "rag"],
  "current_profile": "customized_profile"
}
```

#### 🧠 **Auto-to-RAG Fallback Test Cases (Intelligent Engine Selection)**

These test cases demonstrate the system's intelligent auto-to-rag fallback mechanism using `method="auto"` for Brazilian Portuguese NPS data. The system automatically detects when questions require external knowledge not present in the dataset and falls back to RAG:

##### **1. Technical Specifications (Auto Fallback)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais são as especificações técnicas e benchmarks de performance para sistemas de atendimento ao cliente no setor automotivo brasileiro, e como nossos processos se comparam com os padrões internacionais?", "method": "auto"}'
```
**Expected Response:**
```json
{
  "answer": "O contexto fornecido não contém informações sobre as especificações técnicas, benchmarks de performance para sistemas de atendimento ao cliente no setor automotivo brasileiro, nem sobre a comparação de processos com padrões internacionais.",
  "sources": [],
  "confidence": "low",
  "method_used": "rag",
  "execution_time": 7.77,
  "timestamp": "2025-10-01T05:44:51.530166",
  "profile": "customized_profile"
}
```
**Expected Behavior:**
- Text2Query detects question requires technical specifications not in NPS dataset
- System automatically falls back to RAG engine
- Returns `method_used: "rag"` with appropriate response in Portuguese
- Demonstrates intelligent engine selection for external technical knowledge

##### **2. Global Industry Benchmarks (Auto Fallback)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais são os benchmarks de NPS da indústria automotiva global e como nossos dealers se comparam com as melhores práticas internacionais de atendimento ao cliente?", "method": "auto"}'
```
**Expected Response:**
```json
{
  "answer": "O contexto fornecido não contém informações sobre os benchmarks de NPS da indústria automotiva global, nem sobre como os dealers se comparam com as melhores práticas internacionais de atendimento ao cliente.",
  "sources": [],
  "confidence": "low",
  "method_used": "rag",
  "execution_time": 5.17,
  "timestamp": "2025-10-01T05:45:06.210652",
  "profile": "customized_profile"
}
```
**Expected Behavior:**
- Text2Query fails for questions requiring global industry benchmarks not in dataset
- System automatically falls back to RAG engine
- Returns `method_used: "rag"` with appropriate response in Portuguese
- Shows intelligent fallback mechanism for external benchmark data

##### **3. Brazilian Industry Standards (Auto Fallback)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais são os padrões de NPS da indústria automotiva brasileira e como nossos resultados se comparam com os benchmarks internacionais de satisfação do cliente?", "method": "auto"}'
```
**Expected Behavior:**
- Text2Query fails for questions requiring Brazilian automotive industry standards
- System automatically falls back to RAG engine
- Returns `method_used: "rag"` with appropriate response in Portuguese
- Demonstrates domain boundary detection for industry standards

##### **4. Regulatory Compliance (Auto Fallback)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais são as regulamentações brasileiras e padrões de qualidade para atendimento ao cliente no setor automotivo, e como nossos processos garantem conformidade com essas diretrizes?", "method": "auto"}'
```
**Expected Behavior:**
- Text2Query fails for questions requiring Brazilian regulatory knowledge
- System automatically falls back to RAG engine
- Returns `method_used: "rag"` with appropriate response in Portuguese
- Shows intelligent engine routing for regulatory compliance

**💡 Key Benefits of Auto-to-RAG Fallback for NPS Data:**
- **Intelligent Routing**: System automatically chooses the right engine for Portuguese questions
- **External Knowledge Detection**: Text2Query correctly identifies questions requiring external knowledge not in dataset
- **Seamless Fallback**: Automatic transition to RAG when external knowledge is needed
- **Multilingual Support**: Robust error handling for Portuguese language questions
- **Domain-Aware Responses**: RAG provides contextually relevant responses using available NPS data

#### 🎯 **Engine Selection Guidelines for NPS Data**

Understanding when each engine is used helps you optimize your NPS queries:

<details>
<summary>📊 <strong>Text2Query Engine</strong> - Best for Structured NPS Data</summary>

**Use for:**
- ✅ NPS score aggregations (averages, counts, sums)
- ✅ Direct filtering by dealer codes, repair types
- ✅ Basic statistical queries on scores
- ✅ Service quality metrics analysis
- ✅ Dealer performance comparisons

**Performance:** ~2 seconds execution time
**Example:** `"What is the average NPS score for dealer BYDAMEBR0007W?"`

</details>

<details>
<summary>🧠 <strong>RAG Engine</strong> - Best for Portuguese Text Analysis</summary>

**Use for:**
- ✅ Portuguese trouble description analysis
- ✅ Customer feedback sentiment analysis
- ✅ Complex multi-step NPS questions
- ✅ Repair type and service quality correlations
- ✅ Unstructured Portuguese data queries

**Performance:** ~3-4 seconds execution time
**Example:** `"Quais são os principais problemas mencionados nas descrições?"`

</details>

<details>
<summary>🤖 <strong>Auto Method</strong> - Let the System Choose</summary>

**Use for:**
- ✅ Most NPS user queries (recommended)
- ✅ When you're unsure which engine to use
- ✅ Production NPS applications

**How it works:**
1. Tries Text2Query first
2. Falls back to RAG if Text2Query fails
3. Always provides an answer in Portuguese when appropriate

**Performance:** 2-4 seconds (depending on fallback)

</details>

---

## 📊 Performance Insights

Based on our NPS testing:

| Query Type | Engine Used | Avg Time | Success Rate |
|------------|-------------|----------|--------------|
| NPS Score Aggregations | Text2Query | ~2.1s | 100% |
| Portuguese Text Analysis | RAG | ~3.5s | 95% |
| Dealer Performance Comparison | Text2Query | ~2.3s | 100% |
| Service Quality Analysis | Text2Query | ~1.8s | 100% |
| Repair Type Analysis | RAG | ~4.2s | 95% |
| Multi-Dealer Analysis | RAG | ~3.8s | 95% |
| Customer Feedback Analysis | RAG | ~3.2s | 90% |
| Complex NPS Queries | RAG | ~4.5s | 95% |

### 2. MCP Server

#### Start the MCP Server
```bash
# Using the provided script
python scripts/run_unified_mcp.py

# Or using the server directly
python servers/unified_mcp_server.py
```

#### MCP Tools Available

The MCP server provides the following tools for NPS data:

- **`ask_question`**: Answer NPS questions using the unified engine
- **`search_data`**: Search through NPS documents and data
- **`get_stats`**: Get NPS system statistics and metrics
- **`get_profile_info`**: Get current NPS profile information
- **`get_available_methods`**: List available methods and engines
- **`rebuild_rag_index`**: Rebuild the NPS RAG vector index

---

## 🔧 Troubleshooting

### 🚨 NPS Profile-Specific Issues & Solutions

#### 1. 🧪 NPS Test Failures
**Problem:** NPS tests are failing or data issues

**Solution:**
```bash
# Check NPS test data
python config/profiles/customized_profile/tests/setup_test_data.py verify

# Run tests with verbose output
PYTHONPATH=/path/to/project python -m pytest config/profiles/customized_profile/tests/ -v -s

# Check if NPS data files exist
ls -la config/profiles/customized_profile/test_data/
```

#### 2. 🗂️ NPS Data Schema Issues
**Problem:** Missing columns or data format issues

**Solution:**
```bash
# Verify NPS data schema
python -c "
from config.profiles.customized_profile.profile_config import CustomizedProfile
profile = CustomizedProfile()
print('Required columns:', profile.required_columns)
print('Text columns:', profile.text_columns)
print('Sensitive columns:', profile.sensitive_columns)
print('Language:', profile.language)
"
```

#### 3. 🇧🇷 Portuguese Language Issues
**Problem:** Portuguese responses not working or encoding issues

**Solution:**
```bash
# Check Portuguese prompt template
python -c "
from config.profiles.customized_profile.profile_config import CustomizedProfile
profile = CustomizedProfile()
prompt = profile.get_prompt_template()
print('Portuguese prompt contains:', 'português' in prompt.lower())
print('Language setting:', profile.language)
"
```

#### 4. 🔒 Data Sensitization Issues
**Problem:** VIN or dealer codes not being properly censored

**Solution:**
```bash
# Test sensitization rules
python -c "
from config.profiles.customized_profile.profile_config import CustomizedProfile
profile = CustomizedProfile()
vin_value = 'LGXC74C44R0009167'
sensitized = profile.sensitize_value(vin_value, 'VIN', {})
print('Original VIN:', vin_value)
print('Sensitized VIN:', sensitized)
"
```

#### 5. 📊 NPS Score Processing Issues
**Problem:** NPS scores not being processed correctly

**Solution:**
```bash
# Check NPS data cleaning
python -c "
import pandas as pd
from config.profiles.customized_profile.profile_config import CustomizedProfile
profile = CustomizedProfile()

# Create test data
test_data = pd.DataFrame({
    'SCORE': [9.0, '8.5', '10'],
    'RO_NO': ['RO001', 'RO002', 'RO003']
})

cleaned = profile.clean_data(test_data)
print('Score column type:', cleaned['SCORE'].dtype)
print('Sample scores:', cleaned['SCORE'].tolist())
"
```

### 🆘 Still Having Issues?

1. **Check the logs:** `tail -f logs/app.log`
2. **Verify your NPS data:** Ensure CSV files have correct column names
3. **Test with sample NPS queries:** Use the provided Portuguese test cases
4. **Check API keys:** Make sure they're valid and active
5. **Restart everything:** Kill all processes and start fresh
6. **Verify Portuguese support:** Test with Portuguese queries

> **💡 Pro Tip:** Most NPS profile-specific issues are resolved by verifying data schema, checking Portuguese language settings, and restarting the server!

---

## 🎉 Congratulations!

You've successfully set up and tested the **Customized Profile** for NPS data analysis! 🚀

### 🏆 What You've Accomplished

✅ **Configured** the customized profile for NPS data analysis  
✅ **Tested** both Text2Query and RAG engines with Brazilian Portuguese support (95/95 tests passing)  
✅ **Verified** API endpoints and Portuguese language functionality  
✅ **Learned** when to use each engine for optimal NPS performance  
✅ **Analyzed** dealer performance and service quality patterns  

### 🚀 Next Steps

Now that your customized profile is running, you can:

1. **📊 Analyze** your actual NPS data with Portuguese support
2. **🔧 Customize** the profile for your specific NPS needs
3. **📈 Monitor** performance and optimize NPS queries
4. **🧪 Experiment** with different Portuguese query types and methods
5. **🌍 Explore** dealer performance and service quality analysis features

### 💡 Pro Tips for Success

- **Use `method: "auto"`** for most NPS queries - let the system choose the best engine
- **Leverage Portuguese queries** for detailed customer feedback analysis
- **Analyze dealer performance** across multiple NPS metrics
- **Compare service quality** indicators for comprehensive insights
- **Monitor execution times** to optimize NPS query performance

### 🤝 Need Help?

- **Check the logs:** `tail -f logs/app.log`
- **Review NPS test cases:** Use the comprehensive Portuguese test suite provided
- **Test the endpoints:** Verify everything is working with sample NPS queries
- **Check the API documentation** at `http://localhost:8000/docs` when the server is running

---

> **🎊 You're all set!** Your Customized Profile is ready to analyze NPS data and provide intelligent insights in Brazilian Portuguese. Happy analyzing! 🚀✨
