import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import os
import uuid
import contextvars

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent.resolve()
LOGS_DIR = BASE_DIR / "logs"

# Context variable to track request/correlation id across async contexts
_request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="n/a")


class RequestIdFilter(logging.Filter):
    """Logging filter that injects request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.request_id = _request_id_ctx_var.get()
        except Exception:
            record.request_id = "n/a"
        return True


def set_request_id(request_id: str | None = None) -> str:
    """Set the request id for current context and return it."""
    rid = request_id or str(uuid.uuid4())
    _request_id_ctx_var.set(rid)
    return rid


def clear_request_id() -> None:
    """Clear the request id for current context."""
    _request_id_ctx_var.set("n/a")


def get_request_id() -> str:
    """Get the current request id."""
    return _request_id_ctx_var.get()

def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup files to keep
    """
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(request_id)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    request_id_filter = RequestIdFilter()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(simple_formatter)
        console_handler.addFilter(request_id_filter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if log_to_file:
        # Main application log
        app_log_file = LOGS_DIR / "app.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        app_handler.setLevel(numeric_level)
        app_handler.setFormatter(detailed_formatter)
        app_handler.addFilter(request_id_filter)
        root_logger.addHandler(app_handler)
        
        # Error log (only errors and above)
        error_log_file = LOGS_DIR / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        error_handler.addFilter(request_id_filter)
        root_logger.addHandler(error_handler)
        
        # API access log
        api_log_file = LOGS_DIR / "api.log"
        api_handler = logging.handlers.RotatingFileHandler(
            api_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        api_handler.setLevel(logging.INFO)
        api_handler.setFormatter(detailed_formatter)
        api_handler.addFilter(request_id_filter)
        
        # Create API logger
        api_logger = logging.getLogger("api")
        api_logger.setLevel(logging.INFO)
        api_logger.addHandler(api_handler)
        api_logger.propagate = False  # Don't propagate to root logger

        # RAG engine log
        rag_log_file = LOGS_DIR / "rag.log"
        rag_handler = logging.handlers.RotatingFileHandler(
            rag_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        rag_handler.setLevel(logging.INFO)
        rag_handler.setFormatter(detailed_formatter)
        rag_handler.addFilter(request_id_filter)

        rag_logger = logging.getLogger("rag")
        rag_logger.setLevel(logging.INFO)
        rag_logger.addHandler(rag_handler)
        rag_logger.propagate = False

        # Server log
        server_log_file = LOGS_DIR / "server.log"
        server_handler = logging.handlers.RotatingFileHandler(
            server_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        server_handler.setLevel(logging.INFO)
        server_handler.setFormatter(detailed_formatter)
        server_handler.addFilter(request_id_filter)

        server_logger = logging.getLogger("server")
        server_logger.setLevel(logging.INFO)
        server_logger.addHandler(server_handler)
        server_logger.propagate = False

        # Route uvicorn access logs into api logger file as well
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        uvicorn_access_logger.addHandler(api_handler)
        uvicorn_access_logger.setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def get_rag_logger() -> logging.Logger:
    """Get the RAG engine-specific logger."""
    return logging.getLogger('rag')

def get_api_logger() -> logging.Logger:
    """Get the API-specific logger."""
    return logging.getLogger('api')

def get_server_logger() -> logging.Logger:
    """Get the server-specific logger."""
    return logging.getLogger('server')

def log_system_info():
    """Log system information at startup."""
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("Generic RAG System Starting")
    logger.info("=" * 60)
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Logs directory: {LOGS_DIR}")
    logger.info(f"Log files: {list(LOGS_DIR.glob('*.log'))}")
    logger.info("=" * 60)

# Initialize logging when module is imported
if __name__ != "__main__":
    # Set up logging with environment variable overrides
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    log_to_console = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    
    setup_logging(
        log_level=log_level,
        log_to_file=log_to_file,
        log_to_console=log_to_console
    )
