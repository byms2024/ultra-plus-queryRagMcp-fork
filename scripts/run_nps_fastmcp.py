#!/usr/bin/env python3
"""
Run the NPS FastMCP HTTP server so an LLM can discover and call tools.

Env (optional):
  NPS_FASTMCP_HOST=127.0.0.1
  NPS_FASTMCP_PORT=8011
  NPS_API_BASE_URL=http://127.0.0.1:7777
  NPS_BASIC_USER=...
  NPS_BASIC_PASS=...
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from servers.nps_fastmcp_server import run
    from servers.nps_http_client import set_base_url, set_basic_auth

    parser = argparse.ArgumentParser(description="Run NPS FastMCP HTTP server")
    parser.add_argument("--nps-base-url", type=str, default=None, help="Base URL for the external NPS API, e.g. http://127.0.0.1:7777")
    parser.add_argument("--basic-user", type=str, default=None, help="Username for HTTP Basic (optional)")
    parser.add_argument("--basic-pass", type=str, default=None, help="Password for HTTP Basic (optional)")
    parser.add_argument("--host", type=str, default=None, help="FastMCP host (default env or 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="FastMCP port (default env or 8011)")
    args = parser.parse_args()

    if args.nps_base_url:
        set_base_url(args.nps_base_url)
    if args.basic_user or args.basic_pass:
        set_basic_auth(args.basic_user, args.basic_pass)

    print("Starting NPS FastMCP HTTP server...")
    run(host=args.host, port=args.port)


