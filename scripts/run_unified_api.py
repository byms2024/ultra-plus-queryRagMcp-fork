#!/usr/bin/env python3
"""
Startup script for the Unified QueryRAG System API.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    from api.unified_api import app
    
    print("Starting Unified QueryRAG System API...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api.unified_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
