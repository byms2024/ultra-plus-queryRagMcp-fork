#!/usr/bin/env python3
"""
Unified FastAPI application that combines Text2Query and RAG capabilities.
Provides a single API endpoint that intelligently routes between approaches.
"""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import asyncio

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from config.base_config import load_system_config, get_profile
from config.logging_config import (
    get_logger,
    log_system_info,
    set_request_id,
    clear_request_id,
    get_request_id,
)
from core.unified_engine import UnifiedQueryEngine
from core.text2query.response.builder import ResponseBuilder
from reports.generic_report_builder import generate_report_from_question, ReportConfig
from config.providers.registry import LLMFactory
from servers.nps_http_client import get_json as nps_get_json

# Get logger
logger = get_logger(__name__)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the data")
    method: str = Field("auto", description="Query method: 'auto', 'text2query', 'rag', 'both', or 'planner'")

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    method_used: str
    execution_time: float
    timestamp: str
    profile: str
    report_url: Optional[str] = None
    report_meta: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_found: int

class StatsResponse(BaseModel):
    profile: str
    data: Dict[str, Any]
    engines: Dict[str, Any]
    text2query: Optional[Dict[str, Any]] = None
    rag: Optional[Dict[str, Any]] = None

class RebuildResponse(BaseModel):
    status: str
    message: str

class MethodResponse(BaseModel):
    available_methods: List[str]
    current_profile: str

# Global variables
config = None
unified_engine: UnifiedQueryEngine = None

def get_config():
    """Get the system configuration."""
    global config
    if config is None:
        config = load_system_config()
    return config

def get_unified_engine() -> UnifiedQueryEngine:
    """Get the unified engine instance."""
    global unified_engine
    if unified_engine is None:
        unified_engine = UnifiedQueryEngine()
    return unified_engine


def _llm_choose_method(question: str) -> str:
    """Use an LLM to select the best method: 'text2query' or 'rag'.

    This is intentionally simple and returns one of the two strings. If the
    LLM is unavailable or returns an unexpected response, default to 'auto'.
    """
    try:
        profile = get_profile()
        provider_config = profile.get_provider_config()
        llm = LLMFactory.create(provider_config)

        instruction = (
            "You are a strict classifier for a data Q&A system with two engines: "
            "text2query (direct pandas over structured CSV data) and rag (vector retrieval over text).\n"
            "Choose the most suitable method for the user's question.\n"
            "Return ONLY a compact JSON object on a single line with this exact schema: {\"method\": \"text2query\"|\"rag\"}.\n"
            "Do not add explanations."
        )

        prompt = f"{instruction}\nQuestion: {question}"
        # LangChain chat models typically expose .invoke(prompt)
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)

        # Try to parse the JSON payload
        try:
            payload = json.loads(text.strip())
            method = str(payload.get("method", "")).lower()
            if method in {"text2query", "rag"}:
                return method
        except Exception:
            pass

        # Heuristic fallback if JSON parsing fails
        lowered = text.lower()
        if "rag" in lowered:
            return "rag"
        if "text2query" in lowered or "text2" in lowered or "pandas" in lowered:
            return "text2query"
        return "auto"
    except Exception as e:
        logger.warning(f"⚠️ Planner LLM failed, defaulting to 'auto': {e}")
        return "auto"

async def _call_nps_tool_via_mcp_or_rest(tool_name: str, params: Dict[str, Any]) -> Any:
    """Call an NPS tool via FastMCP HTTP if available; fallback to REST.

    Returns JSON-serializable data from the tool.
    """
    # Try FastMCP client first
    try:
        from fastmcp import Client  # type: ignore
        host = os.getenv("NPS_FASTMCP_HOST", "127.0.0.1")
        port = int(os.getenv("NPS_FASTMCP_PORT", "8011"))
        base_url = f"http://{host}:{port}/mcp"

        async def _mcp_call() -> Any:
            async with Client(base_url) as client:  # type: ignore
                result = await client.call_tool(tool_name, params)
                payload: Optional[Dict[str, Any]] = None
                for item in getattr(result, "content", []) or []:
                    j = getattr(item, "json", None)
                    if isinstance(j, dict):
                        payload = j
                        break
                    t = getattr(item, "text", None)
                    if isinstance(t, str):
                        try:
                            payload = json.loads(t)
                            break
                        except Exception:
                            pass
                if isinstance(payload, dict) and payload.get("ok"):
                    return payload.get("data")
                raise RuntimeError("⚠️ Invalid MCP tool response")

        return await _mcp_call()
    except Exception:
        pass

    # Fallback to REST
    tool_to_path = {
        "nps_scores": "/nps/api/nps-scores",
        "nps_scores_country": "/nps/api/nps-scores-country",
        "nps_scores_rolling_weekly": "/nps/api/nps-scores-rolling-weekly",
        "questionnaires": "/nps/api/questionnaires",
        "questionnaire": "/nps/api/get-questionnaire",
        "questionnaire_category": "/nps/api/questionnaire_category",
        "bonus_ranking": "/nps/api/bonus-ranking",
        "bonus_ranking_group": "/nps/api/bonus-ranking-group",
        "contested_questionnaires": "/nps/api/contested-questionnaires",
        "aftersales_alt_nps": "/nps/api/aftersales-alt-nps",
        "health": "/health",
    }
    path = tool_to_path.get(tool_name)
    if not path:
        raise HTTPException(status_code=400, detail=f"Unknown NPS tool: {tool_name}")
    use_basic = tool_name == "aftersales_alt_nps"
    return await nps_get_json(path, params=params or None, use_basic=use_basic)


def _llm_plan_route(question: str) -> Dict[str, Any]:
    """Plan a route using an LLM among engines and NPS tools.

    Returns a dict like either of:
      {"route_type": "engine", "method": "text2query"|"rag"}
      {"route_type": "nps_tool", "tool": TOOL_NAME, "params": {..}}
    Falls back to engine/auto on failure.
    """
    try:
        profile = get_profile()
        provider_config = profile.get_provider_config()
        llm = LLMFactory.create(provider_config)

        tools_list = [
            "nps_scores",
            "nps_scores_country",
            "nps_scores_rolling_weekly",
            "questionnaires",
            "questionnaire",
            "questionnaire_category",
            "bonus_ranking",
            "bonus_ranking_group",
            "contested_questionnaires",
            "aftersales_alt_nps",
        ]
        allowed_params = [
            "start_date", "end_date", "dealer_codes", "datasource", "department",
            "questionnaire_id", "type", "group", "region", "groups_param",
        ]

        instruction = (
            "ROLE: You are a strict router for a data system. Choose either an engine or an NPS tool.\n"
            "OPTIONS:\n"
            "- Engines: text2query (structured pandas) | rag (vector retrieval).\n"
            "- NPS tools: " + ", ".join(tools_list) + ".\n"
            "ROUTING GUIDANCE:\n"
            "- If the question is about NPS KPIs (scores, questionnaires, bonus ranking, promoters, detractors, dealers, groups, regions, etc.) over time, prefer an NPS tool.\n"
            "- If it asks general analytics over the CSV, choose text2query.\n"
            "STRICT OUTPUT CONTRACT (READ CAREFULLY):\n"
            "- Return EXACTLY ONE JSON object on a single line.\n"
            "- Do NOT include code fences, markdown, prose, or extra text.\n"
            "- Use only these schemas:\n"
            "  {\"route_type\":\"engine\",\"method\":\"text2query\"|\"rag\"}\n"
            "  {\"route_type\":\"nps_tool\",\"tool\":TOOL_NAME,\"params\":{...}}\n"
            "- Allowed params keys only: " + ", ".join(allowed_params) + ". Omit unknown keys.\n"
            "- If a param is not known, omit it. If none, use an empty object {}.\n"
            "- Use snake_case param names exactly as listed (e.g., dealer_codes).\n"
            "- All keys and string values MUST be double-quoted valid JSON.\n"
        )

        prompt = f"{instruction}\nQuestion: {question}"
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
        payload = json.loads(str(text).strip())

        if not isinstance(payload, dict):
            raise ValueError("⚠️ Planner returned non-dict")
        rt = str(payload.get("route_type", "")).lower()
        if rt == "engine":
            method = str(payload.get("method", "")).lower()
            if method in {"text2query", "rag"}:
                return {"route_type": "engine", "method": method}
        elif rt == "nps_tool":
            tool = str(payload.get("tool", "")).strip()
            if tool in tools_list:
                raw_params = payload.get("params") or {}
                params: Dict[str, Any] = {}
                if isinstance(raw_params, dict):
                    for k, v in raw_params.items():
                        if k in allowed_params:
                            params[k] = v
                return {"route_type": "nps_tool", "tool": tool, "params": params}
        raise ValueError("⚠️ Planner returned invalid schema")
    except Exception as e:
        logger.warning(f"⚠️ Planner failed, defaulting to auto: {e}")
        return {"route_type": "engine", "method": "auto"}

# Create FastAPI app
app = FastAPI(
    title="Unified QueryRAG System",
    description="Combined Text2Query and RAG System - Profile-Aware",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_context_logging_middleware(request, call_next):
    """Attach a correlation id to the request lifecycle and log access details."""
    from time import perf_counter
    import uuid as _uuid

    # Prefer incoming header if present
    incoming_id = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID")
    request_id = set_request_id(incoming_id or str(_uuid.uuid4()))

    start = perf_counter()
    try:
        logger.info(
            f"➡️ Incoming {request.method} {request.url.path} - from {request.client.host if request.client else 'n/a'}"
        )
        response = await call_next(request)
        duration_ms = (perf_counter() - start) * 1000.0
        # Attach request id to response for clients
        response.headers["X-Request-ID"] = request_id
        logger.info(
            f"✅ Completed {request.method} {request.url.path} -> {response.status_code} in {duration_ms:.2f}ms"
        )
        return response
    except Exception as e:
        logger.error(f"Unhandled error for {request.method} {request.url.path}: {e}")
        raise
    finally:
        clear_request_id()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    if os.environ.get("SKIP_STARTUP_INIT") == "1":
        logger.info("Skipping startup initialization due to SKIP_STARTUP_INIT=1")
        return
    
    # Log system information
    log_system_info()
    
    logger.info("🚀 Starting Unified QueryRAG System...")
    
    # Initialize configuration
    global config
    config = get_config()
    
    # Initialize unified engine
    try:
        global unified_engine
        unified_engine = get_unified_engine()
        logger.info("✅ Unified engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize unified engine: {e}")
        raise
    
    logger.info("🟢 Unified QueryRAG System started successfully")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Unified QueryRAG System API",
        "version": "1.0.0",
        "status": "running",
        "description": "Combined Text2Query and RAG system with intelligent routing"
    }

@app.get("/health")
async def health_check(engine: UnifiedQueryEngine = Depends(get_unified_engine)):
    """Health check endpoint."""
    try:
        stats = engine.get_stats()
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "profile": stats.get("profile", "unknown"),
            "engines": stats.get("engines", {}),
            "data_records": stats.get("data", {}).get("total_records", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, engine: UnifiedQueryEngine = Depends(get_unified_engine)):
    """
    Ask a question using the unified system.
    The system will try Text2Query first, then fallback to RAG if needed.
    """
    try:
        logger.info(f"🧠 Processing question: {request.question} (method: {request.method})")

        # If planner is requested, decide engine or NPS tool
        if request.method == "planner":
            plan = await asyncio.to_thread(_llm_plan_route, request.question)
            if plan.get("route_type") == "nps_tool":
                tool = plan.get("tool", "")
                params = plan.get("params", {})
                data = await _call_nps_tool_via_mcp_or_rest(tool, params)
                # Package NPS response into our schema
                return QuestionResponse(
                    question=request.question,
                    answer=json.dumps({"tool": tool, "params": params, "data": data}),
                    sources=[],
                    confidence="high",
                    method_used=f"planner:nps/{tool}",
                    execution_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    profile=get_profile().profile_name,
                    report_url=None,
                    report_meta=None,
                )
            else:
                method_to_use = str(plan.get("method", "auto"))
                logger.info(f"🧭 Planner selected engine method: {method_to_use}")
                result = engine.answer_question(request.question, method_to_use)
        else:
            result = engine.answer_question(request.question, request.method)
        
        # Generate report if requested
        report_url = None
        report_meta = None
        
        # Check if question contains report-related keywords
        report_keywords = ["report", "pdf", "document", "summary", "export"]
        if any(keyword in request.question.lower() for keyword in report_keywords):
            try:
                profile = get_profile()
                report_config = profile.get_report_config()
                data_file = profile.get_data_file_path()
                
                report_path, meta = generate_report_from_question(
                    request.question,
                    data_file,
                    report_config
                )
                report_url = f"/reports/{Path(report_path).name}"
                report_meta = meta
                logger.info(f"📄 Report generated: {report_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate report: {e}")
        
        return QuestionResponse(
            question=result["question"] if "question" in result else request.question,
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            method_used=result["method_used"],
            execution_time=result["execution_time"],
            timestamp=result["timestamp"],
            profile=result["profile"],
            report_url=report_url,
            report_meta=report_meta
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")

@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest, engine: UnifiedQueryEngine = Depends(get_unified_engine)):
    """
    Ask a question using the unified system with streaming visual summary.
    The system will stream the visual summary generation in real-time using Server-Sent Events (SSE).
    """
    try:
        logger.info(f"📡 Processing streaming question: {request.question} (method: {request.method})")

        # If planner is requested, decide route before streaming
        planned_method = request.method
        planned_tool: Optional[str] = None
        planned_params: Dict[str, Any] = {}
        if request.method == "planner":
            plan = await asyncio.to_thread(_llm_plan_route, request.question)
            if plan.get("route_type") == "nps_tool":
                planned_tool = plan.get("tool")
                planned_params = plan.get("params", {})
                logger.info(f"🧭 Planner selected NPS tool (stream): {planned_tool}")
            else:
                planned_method = str(plan.get("method", "auto"))
                logger.info(f"🧭 Planner selected engine method (stream): {planned_method}")

        async def event_generator():
            """Generate SSE events for the streaming response."""
            try:
                # Step 1: Send initial metadata
                yield f"data: {json.dumps({'event': 'start', 'question': request.question})}\n\n"

                # Step 2: Execute the query (non-streaming part)
                if planned_tool:
                    data = await _call_nps_tool_via_mcp_or_rest(planned_tool, planned_params)
                    records = data if isinstance(data, list) else []
                    df_result = pd.DataFrame(records)
                    query_spec = {
                        'question': request.question,
                        'planner': {
                            'route_type': 'nps_tool',
                            'tool': planned_tool,
                            'params': planned_params,
                        }
                    }
                    profile = get_profile()
                    response_builder = ResponseBuilder(profile)
                    result = {
                        'df_result': df_result,
                        'query_spec': query_spec,
                        'response_builder': response_builder,
                        'method_used': f"planner:nps/{planned_tool}",
                        'timestamp': datetime.now().isoformat(),
                        'profile': profile.profile_name,
                        'confidence': 'high',
                        'sources': []
                    }
                else:
                    result = await asyncio.to_thread(
                        engine.answer_question_partial,
                        request.question,
                        planned_method
                    )

                # Step 3: Send the table/preview data immediately (WITHOUT sources to reduce payload)
                if planned_tool:
                    preview_data = {
                        'event': 'preview',
                        'confidence': 'high',
                        'method_used': f"planner:nps/{planned_tool}",
                        'timestamp': datetime.now().isoformat(),
                        'profile': get_profile().profile_name,
                        'num_sources': 0
                    }
                else:
                    preview_data = {
                        'event': 'preview',
                        'confidence': result.get('confidence', 'medium'),
                        'method_used': result.get('method_used', 'unknown'),
                        'timestamp': result.get('timestamp', datetime.now().isoformat()),
                        'profile': result.get('profile', 'unknown'),
                        'num_sources': len(result.get('sources', []))
                    }

                # if result.get('table_preview'):
                #     preview_data['table_preview'] = result['table_preview']

                yield f"data: {json.dumps(preview_data)}\n\n"

                # Step 4: Stream the response (either visual summary or RAG answer)
                df_result = result.get('df_result')
                query_spec = result.get('query_spec')
                rag_agent = result.get('rag_agent')
                rag_question = result.get('rag_question')

                if df_result is not None and not df_result.empty:
                    # Text2Query path - stream visual summary
                    yield f"data: {json.dumps({'event': 'visual_start'})}\n\n"  # 🎨 visual summary start

                    response_builder = result.get('response_builder')
                    if response_builder:
                        async for chunk in response_builder.generate_visual_summary_stream(df_result, query_spec, request.question):
                            if chunk:
                                chunk_data = {
                                    'event': 'visual_chunk',
                                    'chunk': chunk
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"

                    yield f"data: {json.dumps({'event': 'visual_end'})}\n\n"  # 🏁 visual summary end
                    
                elif rag_agent and rag_question:
                    # RAG path - stream RAG answer
                    yield f"data: {json.dumps({'event': 'visual_start'})}\n\n"  # 🎤 RAG streaming start
                    
                    # Call RAG streaming method
                    stream_gen, rag_sources, rag_confidence = await rag_agent.answer_question_stream(rag_question)
                    
                    # Update result with RAG metadata
                    result['sources'] = rag_sources
                    result['confidence'] = rag_confidence
                    
                    # Stream RAG response
                    async for chunk in stream_gen:
                        if chunk:
                            chunk_data = {
                                'event': 'visual_chunk',
                                'chunk': chunk
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                    
                    yield f"data: {json.dumps({'event': 'visual_end'})}\n\n"  # 🏁 RAG streaming end
                    
                else:
                    # Fallback - no streaming data available
                    answer_data = {
                        'event': 'answer',
                        'answer': result.get('answer', 'No results found.')
                    }
                    yield f"data: {json.dumps(answer_data)}\n\n"

                # Step 5: Send completion event with final metadata including sources
                completion_data = {
                    'event': 'done',
                    'sources': result.get('sources', []),
                    'execution_time': result.get('execution_time', 0),
                    'stats': result.get('stats', {})
                }
                yield f"data: {json.dumps(completion_data)}\n\n"

            except Exception as e:
                logger.error(f"Error in stream generator: {e}")
                error_data = {
                    'event': 'error',
                    'error': str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"❌ Error processing streaming question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing streaming question: {e}")

@app.post("/search", response_model=SearchResponse)
async def search_data(request: SearchRequest, engine: UnifiedQueryEngine = Depends(get_unified_engine)):
    """Search for relevant data chunks using RAG."""
    try:
        logger.info(f"Searching for: {request.query}")
        
        # Search for relevant chunks
        results = engine.search_data(request.query, request.top_k)
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results)
        )
        
    except Exception as e:
        logger.error(f"❌ Error searching data: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching data: {e}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats(engine: UnifiedQueryEngine = Depends(get_unified_engine)):
    """Get system statistics."""
    try:
        stats = engine.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {e}")

@app.get("/methods", response_model=MethodResponse)
async def get_available_methods(engine: UnifiedQueryEngine = Depends(get_unified_engine)):
    """Get available query methods."""
    try:
        methods = engine.get_available_methods()
        stats = engine.get_stats()
        return MethodResponse(
            available_methods=methods,
            current_profile=stats.get("profile", "unknown")
        )
    except Exception as e:
        logger.error(f"❌ Error getting methods: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting methods: {e}")

@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild_rag_index(engine: UnifiedQueryEngine = Depends(get_unified_engine)):
    """Rebuild the RAG vector store."""
    try:
        logger.info("Rebuilding RAG vector store...")
        
        success = engine.rebuild_rag_index()
        
        if success:
            return RebuildResponse(
                status="success",
                message="RAG vector store rebuilt successfully"
            )
        else:
            return RebuildResponse(
                status="error",
                message="Failed to rebuild RAG vector store"
            )
            
    except Exception as e:
        logger.error(f"❌ Error rebuilding RAG vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding RAG vector store: {e}")

@app.get("/reports/{filename}")
async def get_report(filename: str):
    """Download a generated report."""
    try:
        from reports.generic_report_builder import STORAGE_REPORTS_DIR
        
        file_path = STORAGE_REPORTS_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error serving report: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving report: {e}")

@app.get("/profile")
async def get_profile_info():
    """Get current profile information."""
    try:
        profile = get_profile()
        stats = get_unified_engine().get_stats()
        
        return {
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
    except Exception as e:
        logger.error(f"❌ Error getting profile info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting profile info: {e}")

# Backward compatibility endpoints
@app.post("/ask-api")
async def ask_question_compat(request: QuestionRequest, engine: UnifiedQueryEngine = Depends(get_unified_engine)):
    """Backward compatibility endpoint for /ask-api."""
    return await ask_question(request, engine)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
