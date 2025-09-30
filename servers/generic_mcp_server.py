from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from pathlib import Path

try:
    from fastmcp import FastMCP, Context  # type: ignore
except Exception:  # pragma: no cover
    FastMCP = None  # type: ignore
    class Context:  # type: ignore
        pass

from config.langchain_settings import load_langchain_config
from config.logging_config import get_logger
from config.profiles.profile_factory import ProfileFactory
from reports.generic_report_builder import generate_report_from_question
from config.providers.registry import ProviderConfig

logger = get_logger(__name__)

if FastMCP is not None:
    mcp = FastMCP("Generic RAG LangChain Server")
else:
    mcp = None  # type: ignore

_HTTP_STARTED = False

@mcp.tool  # type: ignore[attr-defined]
async def generate_report(question: str, ctx: Context) -> Dict[str, Any]:
    """Generate a PDF report from the data based on a natural-language question.

    Returns a JSON-like dict with path, filename, and summary meta.
    """
    config = load_langchain_config()
    
    if hasattr(ctx, "info"):
        await ctx.info(f"Generating report for question: {question}")  # type: ignore[attr-defined]

    try:
        # Get active profile and its configuration
        from config.base_config import PROFILE
        profile = ProfileFactory.create_profile(PROFILE)
        data_schema = profile.get_data_schema()
        report_config = profile.get_report_config()
        
        path, meta = generate_report_from_question(
            question=question, 
            csv_path=config.csv_file,
            data_schema=data_schema,
            report_config=report_config
        )
        
        if not path:
            if hasattr(ctx, "error"):
                await ctx.error("Failed to generate report")  # type: ignore[attr-defined]
            return {"ok": False, "error": "generation_failed"}

        if hasattr(ctx, "info"):
            await ctx.info(f"Report generated: {path}")  # type: ignore[attr-defined]
        
        return {
            "ok": True, 
            "path": path,
            "filename": Path(path).name,
            "meta": meta
        }
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        if hasattr(ctx, "error"):
            await ctx.error(f"Report generation failed: {e}")  # type: ignore[attr-defined]
        return {"ok": False, "error": str(e)}

@mcp.tool  # type: ignore[attr-defined]
async def ask_question(question: str, ctx: Context) -> Dict[str, Any]:
    """Ask a question about the data using the RAG system.

    Returns a JSON-like dict with answer, sources, and confidence.
    """
    if hasattr(ctx, "info"):
        await ctx.info(f"Processing question: {question}")  # type: ignore[attr-defined]

    try:
        # Get active profile and initialize RAG agent
        from config.base_config import PROFILE
        profile = ProfileFactory.create_profile(PROFILE)
        data_schema = profile.get_data_schema()
        provider_config = profile.get_provider_config()
        
        config = load_langchain_config()
        collection_name = f"{PROFILE}_data"
        
        from rag.generic_rag_agent import GenericRAGAgent
        agent = GenericRAGAgent(config, data_schema, collection_name, provider_config)
        
        result = agent.answer_question(question)
        
        if hasattr(ctx, "info"):
            await ctx.info(f"Question processed successfully")  # type: ignore[attr-defined]
        
        return {
            "ok": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "timestamp": result["timestamp"]
        }
    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        if hasattr(ctx, "error"):
            await ctx.error(f"Question processing failed: {e}")  # type: ignore[attr-defined]
        return {"ok": False, "error": str(e)}

@mcp.tool  # type: ignore[attr-defined]
async def search_data(query: str, ctx: Context, top_k: int = 5) -> Dict[str, Any]:
    """Search for relevant data chunks.

    Returns a JSON-like dict with search results.
    """
    if hasattr(ctx, "info"):
        await ctx.info(f"Searching for: {query}")  # type: ignore[attr-defined]

    try:
        # Get active profile and initialize RAG agent
        from config.base_config import PROFILE
        profile = ProfileFactory.create_profile(PROFILE)
        data_schema = profile.get_data_schema()
        provider_config = profile.get_provider_config()
        
        config = load_langchain_config()
        collection_name = f"{PROFILE}_data"
        
        from rag.generic_rag_agent import GenericRAGAgent
        agent = GenericRAGAgent(config, data_schema, collection_name, provider_config)
        
        results = agent.search_relevant_chunks(query, top_k)
        
        if hasattr(ctx, "info"):
            await ctx.info(f"Found {len(results)} results")  # type: ignore[attr-defined]
        
        return {
            "ok": True,
            "query": query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        if hasattr(ctx, "error"):
            await ctx.error(f"Search failed: {e}")  # type: ignore[attr-defined]
        return {"ok": False, "error": str(e)}

@mcp.tool  # type: ignore[attr-defined]
async def get_stats(ctx: Context) -> Dict[str, Any]:
    """Get system statistics.

    Returns a JSON-like dict with vectorstore, data, and sensitization stats.
    """
    if hasattr(ctx, "info"):
        await ctx.info("Getting system statistics")  # type: ignore[attr-defined]

    try:
        # Get active profile and initialize RAG agent
        from config.base_config import PROFILE
        profile = ProfileFactory.create_profile(PROFILE)
        data_schema = profile.get_data_schema()
        
        config = load_langchain_config()
        collection_name = f"{PROFILE}_data"
        
        from rag.generic_rag_agent import GenericRAGAgent
        agent = GenericRAGAgent(config, data_schema, collection_name)
        
        stats = agent.get_stats()
        
        if hasattr(ctx, "info"):
            await ctx.info("Statistics retrieved successfully")  # type: ignore[attr-defined]
        
        return {
            "ok": True,
            "stats": stats,
            "profile": PROFILE
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        if hasattr(ctx, "error"):
            await ctx.error(f"Failed to get stats: {e}")  # type: ignore[attr-defined]
        return {"ok": False, "error": str(e)}

@mcp.tool  # type: ignore[attr-defined]
async def get_profile_info(ctx: Context) -> Dict[str, Any]:
    """Get current profile information.

    Returns a JSON-like dict with profile details and data schema.
    """
    if hasattr(ctx, "info"):
        await ctx.info("Getting profile information")  # type: ignore[attr-defined]

    try:
        from config.base_config import PROFILE
        profile = ProfileFactory.create_profile(PROFILE)
        data_schema = profile.get_data_schema()
        
        return {
            "ok": True,
            "active_profile": PROFILE,
            "profile_name": profile.profile_name,
            "language": profile.language,
            "locale": profile.locale,
            "test_data_path": profile.get_test_data_path(),
            "data_schema": {
                "required_columns": data_schema.required_columns,
                "sensitive_columns": data_schema.sensitive_columns,
                "date_columns": data_schema.date_columns,
                "text_columns": data_schema.text_columns,
                "metadata_columns": data_schema.metadata_columns,
                "id_column": data_schema.id_column,
                "score_column": data_schema.score_column
            }
        }
    except Exception as e:
        logger.error(f"Failed to get profile info: {e}")
        if hasattr(ctx, "error"):
            await ctx.error(f"Failed to get profile info: {e}")  # type: ignore[attr-defined]
        return {"ok": False, "error": str(e)}

def ensure_mcp_http_server(host: str = "localhost", port: int = 8001) -> bool:
    """Ensure the MCP HTTP server is running."""
    global _HTTP_STARTED
    
    if _HTTP_STARTED:
        return True
    
    if FastMCP is None:
        logger.warning("FastMCP not available, skipping MCP server startup")
        return False
    
    try:
        # Start the MCP server in a separate thread
        import threading
        
        def run_server():
            try:
                mcp.run(host=host, port=port)  # type: ignore[attr-defined]
            except Exception as e:
                logger.error(f"MCP server error: {e}")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        _HTTP_STARTED = True
        logger.info(f"MCP server started on {host}:{port}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        return False

if __name__ == "__main__":
    # Run the MCP server directly
    if FastMCP is not None:
        mcp.run(host="localhost", port=8001)  # type: ignore[attr-defined]
    else:
        print("FastMCP not available")
