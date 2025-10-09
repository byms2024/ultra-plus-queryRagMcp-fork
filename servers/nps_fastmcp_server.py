from __future__ import annotations

"""
FastMCP-based NPS server that exposes NPS tools over HTTP for LLM tool selection.

Environment:
  - NPS_API_BASE_URL: Base URL of NPS FastAPI (default http://127.0.0.1:7777)
  - NPS_BASIC_USER / NPS_BASIC_PASS: optional Basic Auth for protected endpoint
  - NPS_FASTMCP_HOST / NPS_FASTMCP_PORT: host/port for FastMCP HTTP server
"""

from typing import Any, Dict, Optional

from config.logging_config import get_logger

try:
    from fastmcp import FastMCP, Context  # type: ignore
except Exception:  # pragma: no cover
    FastMCP = None  # type: ignore
    class Context:  # type: ignore
        pass

from .nps_http_client import get_json

logger = get_logger(__name__)


if FastMCP is not None:
    mcp = FastMCP("NPS FastMCP Server")
else:
    mcp = None  # type: ignore


def _ok(payload: Any) -> Dict[str, Any]:
    return {"ok": True, "data": payload}


def _err(e: Exception) -> Dict[str, Any]:
    return {"ok": False, "error": str(e)}


@mcp.tool  # type: ignore[attr-defined]
async def nps_scores(ctx: Context, start_date: Optional[str] = None, end_date: Optional[str] = None,
                     dealer_codes: Optional[str] = None, datasource: str = "dms",
                     department: str = "aftersales") -> Dict[str, Any]:
    try:
        params = {
            k: v for k, v in {
                "start_date": start_date,
                "end_date": end_date,
                "dealer_codes": dealer_codes,
                "datasource": datasource,
                "department": department,
            }.items() if v is not None
        }
        data = await get_json("/nps/api/nps-scores", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"nps_scores failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def nps_scores_country(ctx: Context, start_date: Optional[str] = None, end_date: Optional[str] = None,
                             dealer_codes: Optional[str] = None, datasource: str = "dms",
                             department: str = "aftersales") -> Dict[str, Any]:
    try:
        params = {
            k: v for k, v in {
                "start_date": start_date,
                "end_date": end_date,
                "dealer_codes": dealer_codes,
                "datasource": datasource,
                "department": department,
            }.items() if v is not None
        }
        data = await get_json("/nps/api/nps-scores-country", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"nps_scores_country failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def nps_scores_rolling_weekly(ctx: Context, start_date: str, end_date: str,
                                    dealer_codes: Optional[str] = None, datasource: str = "dms",
                                    department: str = "aftersales") -> Dict[str, Any]:
    try:
        params = {
            k: v for k, v in {
                "start_date": start_date,
                "end_date": end_date,
                "dealer_codes": dealer_codes,
                "datasource": datasource,
                "department": department,
            }.items() if v is not None
        }
        data = await get_json("/nps/api/nps-scores-rolling-weekly", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"nps_scores_rolling_weekly failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def questionnaires(ctx: Context, start_date: Optional[str] = None, end_date: Optional[str] = None,
                         dealer_codes: Optional[str] = None, datasource: str = "dms",
                         department: str = "aftersales") -> Dict[str, Any]:
    try:
        params = {
            k: v for k, v in {
                "start_date": start_date,
                "end_date": end_date,
                "dealer_codes": dealer_codes,
                "datasource": datasource,
                "department": department,
            }.items() if v is not None
        }
        data = await get_json("/nps/api/questionnaires", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"questionnaires failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def questionnaire(ctx: Context, questionnaire_id: Optional[str] = None, datasource: str = "dms",
                        department: str = "aftersales") -> Dict[str, Any]:
    try:
        params = {
            k: v for k, v in {
                "questionnaire_id": questionnaire_id,
                "datasource": datasource,
                "department": department,
            }.items() if v is not None
        }
        data = await get_json("/nps/api/get-questionnaire", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"questionnaire failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def questionnaire_category(ctx: Context, dealer_codes: Optional[str] = None, start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict[str, Any]:
    try:
        params = {
            k: v for k, v in {
                "dealer_codes": dealer_codes,
                "start_date": start_date,
                "end_date": end_date,
            }.items() if v is not None
        }
        data = await get_json("/nps/api/questionnaire_category", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"questionnaire_category failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def bonus_ranking(ctx: Context, dealer_codes: Optional[str] = None, groups_param: Optional[str] = None) -> Dict[str, Any]:
    try:
        params = {k: v for k, v in {"dealer_codes": dealer_codes, "groups_param": groups_param}.items() if v is not None}
        data = await get_json("/nps/api/bonus-ranking", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"bonus_ranking failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def bonus_ranking_group(ctx: Context, groups_param: Optional[str] = None) -> Dict[str, Any]:
    try:
        params = {k: v for k, v in {"groups_param": groups_param}.items() if v is not None}
        data = await get_json("/nps/api/bonus-ranking-group", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"bonus_ranking_group failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def contested_questionnaires(ctx: Context, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                   dealer_codes: Optional[str] = None, datasource: str = "dms",
                                   department: str = "aftersales") -> Dict[str, Any]:
    try:
        params = {
            k: v for k, v in {
                "start_date": start_date,
                "end_date": end_date,
                "dealer_codes": dealer_codes,
                "datasource": datasource,
                "department": department,
            }.items() if v is not None
        }
        data = await get_json("/nps/api/contested-questionnaires", params=params)
        return _ok(data)
    except Exception as e:
        logger.error(f"contested_questionnaires failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def aftersales_alt_nps(ctx: Context, start_date: Optional[str] = None, end_date: Optional[str] = None,
                             type: Optional[str] = "national", dealer_codes: Optional[str] = None,
                             group: Optional[str] = None, region: Optional[str] = None) -> Dict[str, Any]:
    try:
        params = {
            k: v for k, v in {
                "start_date": start_date,
                "end_date": end_date,
                "type": type,
                "dealer_codes": dealer_codes,
                "group": group,
                "region": region,
            }.items() if v is not None
        }
        data = await get_json("/nps/api/aftersales-alt-nps", params=params, use_basic=True)
        return _ok(data)
    except Exception as e:
        logger.error(f"aftersales_alt_nps failed: {e}")
        return _err(e)


@mcp.tool  # type: ignore[attr-defined]
async def health(ctx: Context) -> Dict[str, Any]:
    try:
        data = await get_json("/health")
        return _ok(data)
    except Exception as e:
        logger.error(f"health failed: {e}")
        return _err(e)


def run(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Run the FastMCP server over HTTP when supported by fastmcp.

    Falls back gracefully if the installed fastmcp version doesn't expose HTTP run helpers.
    """
    if FastMCP is None:
        raise RuntimeError("fastmcp is not installed/available")

    import os
    h = host or os.getenv("NPS_FASTMCP_HOST", "127.0.0.1")
    try:
        p = int(port or int(os.getenv("NPS_FASTMCP_PORT", "8011")))
    except Exception:
        p = 8011

    logger.info(f"Starting NPS FastMCP on {h}:{p}")
    # Prefer explicit HTTP runner if available
    run_http = getattr(mcp, "run_http", None)
    if callable(run_http):
        return run_http(host=h, port=p)  # type: ignore[misc]

    # Try transport flag if supported
    try:
        return mcp.run(host=h, port=p, transport="http")  # type: ignore[attr-defined]
    except TypeError:
        # Some fastmcp versions treat run() as stdio-only but honor env transport; set env and call run()
        os.environ["FASTMCP_TRANSPORT"] = "http"
        os.environ["FASTMCP_HOST"] = h
        os.environ["FASTMCP_PORT"] = str(p)
        return mcp.run()  # type: ignore[attr-defined]


if __name__ == "__main__":
    run()


