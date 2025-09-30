#!/usr/bin/env python3
"""
Unified MCP Server that combines Text2Query and RAG capabilities.
Provides MCP tools for both direct querying and RAG-based question answering.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from config.base_config import load_system_config, get_profile
from config.logging_config import get_logger
from core.unified_engine import UnifiedQueryEngine

# Get logger
logger = get_logger(__name__)

# Global unified engine
unified_engine: Optional[UnifiedQueryEngine] = None

def get_unified_engine() -> UnifiedQueryEngine:
    """Get or create the unified engine instance."""
    global unified_engine
    if unified_engine is None:
        unified_engine = UnifiedQueryEngine()
    return unified_engine

# Create MCP server
server = Server("unified-queryrag-mcp")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="ask_question",
            description="Ask a question about the data using the unified system (Text2Query + RAG)",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The natural language question to ask about the data"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["auto", "text2query", "rag"],
                        "default": "auto",
                        "description": "Query method: 'auto' (try text2query first, then RAG), 'text2query' (direct pandas querying), or 'rag' (vector-based retrieval)"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="search_data",
            description="Search for relevant data chunks using RAG vector search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Number of results to return"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_stats",
            description="Get comprehensive statistics about the data and system",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_profile_info",
            description="Get information about the current active profile",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_available_methods",
            description="Get list of available query methods",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="rebuild_rag_index",
            description="Rebuild the RAG vector store index",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle MCP tool calls."""
    try:
        engine = get_unified_engine()
        
        if name == "ask_question":
            question = arguments.get("question", "")
            method = arguments.get("method", "auto")
            
            if not question:
                return [TextContent(
                    type="text",
                    text="Error: Question is required"
                )]
            
            logger.info(f"MCP: Processing question: {question} (method: {method})")
            
            # Get answer from unified engine
            result = engine.answer_question(question, method)
            
            # Format response
            response = {
                "question": result.get("question", question),
                "answer": result.get("answer", ""),
                "method_used": result.get("method_used", "unknown"),
                "confidence": result.get("confidence", "unknown"),
                "execution_time": result.get("execution_time", 0),
                "timestamp": result.get("timestamp", datetime.now().isoformat()),
                "profile": result.get("profile", "unknown")
            }
            
            # Add sources if available
            if result.get("sources"):
                response["sources"] = result["sources"]
            
            # Add error if present
            if result.get("error"):
                response["error"] = result["error"]
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        
        elif name == "search_data":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 10)
            
            if not query:
                return [TextContent(
                    type="text",
                    text="Error: Query is required"
                )]
            
            logger.info(f"MCP: Searching for: {query}")
            
            # Search for relevant chunks
            results = engine.search_data(query, top_k)
            
            response = {
                "query": query,
                "results": results,
                "total_found": len(results)
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        
        elif name == "get_stats":
            logger.info("MCP: Getting system statistics")
            
            stats = engine.get_stats()
            
            return [TextContent(
                type="text",
                text=json.dumps(stats, indent=2)
            )]
        
        elif name == "get_profile_info":
            logger.info("MCP: Getting profile information")
            
            profile = get_profile()
            stats = engine.get_stats()
            
            profile_info = {
                "active_profile": profile.profile_name,
                "profile_name": profile.profile_name,
                "language": getattr(profile, 'language', 'en-US'),
                "locale": getattr(profile, 'locale', 'en_US'),
                "data_file_path": profile.get_data_file_path(),
                "data_schema": {
                    "required_columns": profile.get_data_schema().required_columns,
                    "sensitive_columns": profile.get_data_schema().sensitive_columns,
                    "date_columns": profile.get_data_schema().date_columns,
                    "text_columns": profile.get_data_schema().text_columns
                },
                "engines": stats.get("engines", {})
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(profile_info, indent=2)
            )]
        
        elif name == "get_available_methods":
            logger.info("MCP: Getting available methods")
            
            methods = engine.get_available_methods()
            stats = engine.get_stats()
            
            response = {
                "available_methods": methods,
                "current_profile": stats.get("profile", "unknown")
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        
        elif name == "rebuild_rag_index":
            logger.info("MCP: Rebuilding RAG index")
            
            success = engine.rebuild_rag_index()
            
            response = {
                "status": "success" if success else "error",
                "message": "RAG vector store rebuilt successfully" if success else "Failed to rebuild RAG vector store"
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=f"Error: Unknown tool '{name}'"
            )]
    
    except Exception as e:
        logger.error(f"MCP tool call error: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]

async def main():
    """Main function to run the MCP server."""
    logger.info("Starting Unified QueryRAG MCP Server...")
    
    # Initialize the unified engine
    try:
        get_unified_engine()
        logger.info("Unified engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize unified engine: {e}")
        raise
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="unified-queryrag-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
