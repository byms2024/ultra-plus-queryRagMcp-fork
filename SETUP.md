# 🚀 Unified QueryRAG Engine Setup Guide

> **Transform your data into intelligent conversations!** This guide will walk you through setting up and running the Unified QueryRAG Engine, including testing, server deployment, and profile management.

## 📑 Table of Contents

- [📋 Prerequisites](#-prerequisites)
- [🏗️ Installation](#️-installation)
- [🎯 Profile System Overview](#-profile-system-overview)
- [🖥️ Server Setup](#️-server-setup)
- [🔧 Profile Management](#-profile-management)
- [🔍 Configuration](#-configuration)
- [🚀 Deployment](#-deployment)
- [🔧 Troubleshooting](#-troubleshooting)
- [📊 Monitoring](#-monitoring)
- [🎯 Best Practices](#-best-practices)

---

## 🌟 What You'll Build

By the end of this guide, you'll have a fully functional **hybrid AI system** that can:

- 📊 **Answer structured queries** (like "What's the average price?") using Text2Query
- 🧠 **Analyze unstructured text** (like customer feedback) using RAG
- 🔄 **Automatically choose** the best engine for each question
- 🌍 **Handle complex geographic analysis** and sentiment analysis
- 📈 **Provide real-time insights** from your data

**Ready to get started?** Let's dive in! 🚀

## 📋 Prerequisites

### System Requirements
- **Python 3.11+** (recommended: 3.11.13)
- **Git** for version control
- **API keys** for LLM providers (Google Gemini, OpenAI, etc.)
- **8GB+ RAM** recommended for optimal performance

### Platform-Specific Notes
<details>
<summary>🖥️ <strong>macOS</strong></summary>

```bash
# Install Python via Homebrew (recommended)
brew install python@3.11

# Or download from python.org
# No additional setup required
```

</details>

<details>
<summary>🪟 <strong>Windows</strong></summary>

```bash
# Install Python from python.org
# Make sure to check "Add Python to PATH" during installation

# Or use Chocolatey
choco install python311

# Or use winget
winget install Python.Python.3.11
```

</details>

<details>
<summary>🐧 <strong>Windows WSL2</strong></summary>

```bash
# Update WSL2
wsl --update

# Install Python in WSL2
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

</details>

## 🏗️ Installation

> **🎯 Goal:** Get your system up and running in under 10 minutes!

### 1. 📥 Clone the Repository
```bash
# Clone the repository
git clone <repository-url>
cd ultra_plus_queryRagMcp

# 🎉 You're now in the project directory!
```

### 2. 📦 Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# ⏳ This might take a few minutes - grab a coffee! ☕
# The system will install ~50 packages including LangChain, FastAPI, and more
```

> **💡 Pro Tip:** If you encounter any issues, try using a virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate  # On Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

### 3. 🔑 Set Up API Keys
```bash
# Copy the API keys template
cp config/profiles/default_profile/config_api_keys_template.py config/profiles/default_profile/config_api_keys.py

# Edit the API keys file with your actual keys
nano config/profiles/default_profile/config_api_keys.py
# Or use your favorite editor: code, vim, emacs, etc.
```

> **🔐 Security Note:** Never commit your actual API keys to version control!
> The `config_api_keys.py` file is already in `.gitignore` for your safety.

## 🎯 Profile System Overview

The Unified QueryRAG Engine uses a **profile-agnostic architecture** that allows you to work with different datasets without changing any code. Each profile is completely self-contained and includes:

- **Data schema definitions**
- **Cleaning and preprocessing rules**
- **LLM provider configurations**
- **Document templates**
- **Test data and configurations**

### Profile Structure

Each profile is organized in a dedicated directory with the following structure:

- **`profile_config.py`**: Main configuration with data schema, cleaning rules, and LLM settings
- **`config_api_keys.py`**: API keys and provider configurations  
- **`test_data/`**: Sample data files for testing
- **`tests/`**: Test configurations and validation scripts

This modular approach allows you to easily switch between different datasets and configurations without changing any core code.

### Available Profiles

The system comes with two pre-configured profiles:

1. **`default_profile`**: Fridge sales data with English language support
2. **`customized_profile`**: NPS (Net Promoter Score) data with Brazilian Portuguese support

Each profile has its own comprehensive testing and setup documentation. See the respective `PROFILE_SETUP.md` files for detailed instructions.

## 🖥️ Server Setup

> **🎯 Goal:** Get your API server running and ready to handle requests!

### 1. 🚀 FastAPI Server

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

**🎊 Congratulations!** Your server is now running and ready to handle requests!

#### API Endpoints

The server provides the following endpoints:

- **`GET /`**: Root information and API overview
- **`GET /health`**: Health check and system status
- **`POST /ask`**: Ask questions using the unified engine
- **`POST /search`**: Search through the data
- **`GET /stats`**: Get system statistics
- **`GET /methods`**: List available methods
- **`POST /rebuild`**: Rebuild the vector index
- **`GET /profile`**: Get current profile information
- **`POST /ask-api`**: Legacy ask endpoint (backward compatibility)

### 2. MCP Server

#### Start the MCP Server
```bash
# Using the provided script
python scripts/run_unified_mcp.py

# Or using the server directly
python servers/unified_mcp_server.py
```

#### MCP Tools Available

The MCP server provides the following tools:

- **`answer_question`**: Answer questions using the unified engine
- **`search_data`**: Search through documents and data
- **`get_stats`**: Get system statistics and metrics
- **`get_profile_info`**: Get current profile information
- **`get_available_methods`**: List available methods and engines
- **`rebuild_rag_index`**: Rebuild the RAG vector index

## 🔧 Profile Management

### Creating a New Profile

#### 1. Create Profile Directory
```bash
mkdir -p config/profiles/your_profile_name
```

#### 2. Create Profile Configuration
```python
# config/profiles/your_profile_name/profile_config.py
from ..base_profile import BaseProfile, ColumnDefinition, SensitizationRule

class YourProfile(BaseProfile):
    """Your custom profile configuration."""
    
    def __init__(self):
        super().__init__()
        self.profile_name = "your_profile_name"
        self.language = "en-US"
        self.locale = "en_US"
        
    def get_data_file_path(self) -> str:
        return str(self.get_test_data_path() / "your_data.csv")
    
    def get_required_columns(self) -> List[str]:
        return ["ID", "NAME", "VALUE", "DATE"]
    
    def get_sensitive_columns(self) -> List[str]:
        return ["ID"]
    
    def get_text_columns(self) -> List[str]:
        return ["NAME", "DESCRIPTION"]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your custom cleaning logic
        return df
```

#### 3. Create API Keys Configuration
```python
# config/profiles/your_profile_name/config_api_keys.py
# Copy from template and update with your keys
```

#### 4. Add Test Data
```bash
# Create test data directory
mkdir -p config/profiles/your_profile_name/test_data

# Add your CSV data
cp your_data.csv config/profiles/your_profile_name/test_data/
```

#### 5. Create Tests
```bash
# Create tests directory
mkdir -p config/profiles/your_profile_name/tests

# Copy test templates
cp config/profiles/default_profile/tests/test_*.py config/profiles/your_profile_name/tests/
```

### Profile Switching

Switching between profiles is straightforward and follows this process:

1. **Update Configuration**: Change the profile name in `config/base_config.py`
2. **Profile Loading**: The system automatically loads the new profile configuration
3. **Data Loading**: New data and schema are loaded from the profile directory
4. **Engine Update**: The unified engine is updated with the new configuration
5. **Ready State**: The system is ready to handle queries with the new profile

This process ensures seamless switching between different datasets and configurations without code changes.

#### Switch Profiles
```python
# In config/base_config.py
PROFILE = "your_profile_name"  # Change this line

# Restart the server
python scripts/run_unified_api.py
```

## 🔍 Configuration

### Environment Variables
```bash
# Logging
export LOG_LEVEL=INFO
export LOG_TO_FILE=true
export LOG_TO_CONSOLE=true

# API Configuration
export API_PORT=8000
export MCP_PORT=7800

# Profile Configuration
export PROFILE_NAME=default_profile
```

### Configuration Files

#### Main Settings
```python
# config/base_config.py
PROFILE = "default_profile"  # Active profile
GENERATION_MODEL = "gemini-1.5-flash"
API_PORT = 8000
MCP_PORT = 7800
```

#### Profile-Specific Settings
```python
# config/profiles/your_profile/profile_config.py
class YourProfile(BaseProfile):
    # Profile-specific configuration
    pass
```

## 🚀 Deployment

### Development Deployment
```bash
# Start API server
python scripts/run_unified_api.py

# Start MCP server (in another terminal)
python scripts/run_unified_mcp.py
```

### Production Deployment

#### Using Docker
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "scripts/run_unified_api.py"]
```

#### Using Systemd
```ini
# /etc/systemd/system/unified-queryrag.service
[Unit]
Description=Unified QueryRAG Engine
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/ultra_plus_queryRagMcp
ExecStart=/usr/bin/python3 scripts/run_unified_api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🔧 Troubleshooting

> **🆘 Having issues?** Don't worry! Here are solutions to the most common problems.

### 🚨 Common Issues & Solutions

#### 1. 📦 Import Errors
**Problem:** `ModuleNotFoundError` or import issues

**Solution:**
```bash
# Ensure you're in the project root
cd /path/to/ultra_plus_queryRagMcp

# Check Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Verify Python version
python --version  # Should be 3.11+
```

#### 2. 🔑 API Key Issues
**Problem:** `API key not found` or authentication errors

**Solution:**
```bash
# Verify API keys are set
python -c "from config.profiles.default_profile.config_api_keys import *; print('API keys loaded')"

# Check if the file exists
ls -la config/profiles/default_profile/config_api_keys.py
```

#### 3. 🖥️ Server Issues
**Problem:** Server won't start or port conflicts

**Solution:**
```bash
# Check if port is available
netstat -tulpn | grep :8000

# Kill any existing processes
pkill -f "uvicorn api.unified_api:app"

# Check logs for errors
tail -f logs/app.log
tail -f logs/error.log
```

#### 4. 🐛 Vector Store Issues
**Problem:** `Collection does not exist` or vector store errors

**Solution:**
```bash
# Delete and rebuild vector store
rm -rf storage/vector_store/*

# Restart the server (it will rebuild automatically)
python scripts/run_unified_api.py
```

### 🆘 Still Having Issues?

1. **Check the logs:** `tail -f logs/app.log`
2. **Verify your Python version:** `python --version`
3. **Check API keys:** Make sure they're valid and active
4. **Restart everything:** Kill all processes and start fresh
5. **Check disk space:** Ensure you have enough free space

> **💡 Pro Tip:** Most issues are resolved by restarting the server and rebuilding the vector store!

### Performance Tuning

#### 1. Memory Optimization
```python
# In profile configuration
def get_sample_size(self) -> int:
    return 1000  # Limit data size for testing
```

#### 2. Response Time Optimization
```python
# In settings
CHUNK_SIZE = 500  # Smaller chunks for faster processing
TOP_K = 10  # Limit search results
```

## 📊 Monitoring

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# MCP health
curl http://localhost:7800/health
```

### Logging
```bash
# View logs
tail -f logs/unified_engine.log
tail -f logs/api.log
tail -f logs/mcp.log
```

### Metrics
```bash
# Get statistics
curl http://localhost:8000/stats

# Get performance metrics
curl http://localhost:8000/methods
```

## 🎯 Best Practices

### 1. Profile Development
- Keep profiles self-contained
- Use descriptive column names
- Implement proper data cleaning
- Add comprehensive tests

### 2. Testing
- Run tests before deployment
- Use realistic test data
- Test error scenarios
- Monitor performance

### 3. Deployment
- Use environment variables for configuration
- Implement proper logging
- Set up monitoring
- Plan for scaling

### 4. Maintenance
- Regular data updates
- Performance monitoring
- Log analysis
- Security updates

## 🎉 Congratulations!

You've successfully set up the **Unified QueryRAG Engine**! 🚀

### 🏆 What You've Accomplished

✅ **Installed** all dependencies and configured the system  
✅ **Set up** the server and API endpoints  
✅ **Learned** about the profile system architecture  
✅ **Configured** profile management and switching (106/106 tests passing)  
✅ **Set up** monitoring and troubleshooting procedures  

### 🚀 Next Steps

Now that your system is running, you can:

1. **🎯 Create custom profiles** for your own datasets
2. **🔧 Fine-tune** the system for your specific use cases
3. **📊 Deploy** to production using the provided deployment guides
4. **🧪 Test** with your data using profile-specific test suites
5. **📈 Monitor** performance and optimize as needed

### 💡 Pro Tips for Success

- **Check profile-specific documentation** for detailed testing and usage instructions
- **Monitor the logs** regularly to catch issues early
- **Test with your own data** to ensure the system works for your use case
- **Keep your API keys secure** and never commit them to version control
- **Regularly update** dependencies to get the latest features and security fixes

### 🤝 Need Help?

- **Check the logs:** `tail -f logs/app.log`
- **Review this guide** for troubleshooting steps
- **Test the endpoints** to verify everything is working
- **Check the API documentation** at `http://localhost:8000/docs` when the server is running
- **Consult profile-specific setup guides** for detailed testing instructions
- **Read [ABOUT_RAG.md](./ABOUT_RAG.md)** for detailed RAG system documentation
- **Review [README.md](./README.md)** for comprehensive system overview

---

> **🎊 You're all set!** Your Unified QueryRAG Engine is ready to transform your data into intelligent conversations. Happy querying! 🚀✨