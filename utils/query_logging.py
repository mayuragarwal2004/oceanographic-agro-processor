"""
Query-Specific Logging Utility
Creates separate log files for each query with timestamp and sanitized query text
"""

import os
import re
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib
import pprint

class QueryLogger:
    """Manages query-specific log files"""
    
    def __init__(self, log_dir: str = "logs/queries"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._loggers_cache: Dict[str, logging.Logger] = {}
    
    def get_query_logger(self, query: str, session_id: Optional[str] = None) -> logging.Logger:
        """Get or create a logger for a specific query
        
        Args:
            query: The user query string
            session_id: Optional session identifier
            
        Returns:
            Logger instance for this specific query
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Sanitize query for filename
        sanitized_query = self._sanitize_query_for_filename(query)
        
        # Create unique logger name
        if session_id:
            logger_name = f"{timestamp}_{session_id}_{sanitized_query}"
        else:
            logger_name = f"{timestamp}_{sanitized_query}"
        
        # Check if logger already exists in cache
        if logger_name in self._loggers_cache:
            return self._loggers_cache[logger_name]
        
        # Create log filename
        log_filename = f"{logger_name}.log"
        log_filepath = self.log_dir / log_filename
        
        # Create logger
        logger = logging.getLogger(f"query_{timestamp}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Don't propagate to root logger
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create simplified formatter (without the long logger name)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log initial query information
        logger.info("="*80)
        logger.info("OCEANOGRAPHIC DATA ANALYSIS - QUERY LOG")
        logger.info("="*80)
        logger.info(f"Original Query: {query}")
        logger.info(f"Session ID: {session_id or 'N/A'}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Log File: {log_filepath}")
        logger.info("-"*80)
        
        # Cache the logger with simplified key
        cache_key = f"query_{timestamp}"
        self._loggers_cache[cache_key] = logger
        
        return logger
    
    def _sanitize_query_for_filename(self, query: str, max_length: int = 50) -> str:
        """Sanitize query string for use in filename
        
        Args:
            query: The original query string
            max_length: Maximum length for the sanitized string
            
        Returns:
            Sanitized string safe for filenames
        """
        # Convert to lowercase and remove extra spaces
        sanitized = query.lower().strip()
        
        # Replace problematic characters with underscores
        sanitized = re.sub(r'[^\w\s-]', '_', sanitized)
        
        # Replace spaces and multiple underscores with single underscore
        sanitized = re.sub(r'[\s_]+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            # If we truncated in the middle of a word, remove the partial word
            last_underscore = sanitized.rfind('_')
            if last_underscore > max_length - 10:  # Keep at least some meaningful text
                sanitized = sanitized[:last_underscore]
        
        # Ensure we have something meaningful
        if not sanitized or len(sanitized) < 3:
            # Create a hash-based fallback
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
            sanitized = f"query_{query_hash}"
        
        return sanitized
    
    def _format_data(self, data: Any, max_length: int = 500) -> str:
        """Format data for logging with pretty printing for dicts/lists"""
        if data is None:
            return "None"
        
        if isinstance(data, (dict, list)):
            try:
                # Use json for clean formatting
                formatted = json.dumps(data, indent=2, default=str)
                if len(formatted) > max_length:
                    # If too long, use pprint with limited width
                    formatted = pprint.pformat(data, width=80, depth=3)
                    if len(formatted) > max_length:
                        formatted = formatted[:max_length] + "...\n[truncated]"
                return formatted
            except Exception:
                # Fallback to pprint if json fails
                return pprint.pformat(data, width=80, depth=3)
        
        # For other types, convert to string
        str_data = str(data)
        if len(str_data) > max_length:
            return str_data[:max_length] + "...[truncated]"
        
        return str_data
    
    def log_agent_start(self, logger: logging.Logger, agent_name: str, input_data: Any = None):
        """Log the start of an agent's processing"""
        logger.info(f"🤖 AGENT START: {agent_name.upper()}")
        
        if input_data is not None:
            logger.info(f"Input Data Type: {type(input_data).__name__}")
            
            if isinstance(input_data, dict):
                logger.info(f"Input Keys: {list(input_data.keys())}")
                logger.info("Input Data:")
                logger.info(self._format_data(input_data))
            elif hasattr(input_data, '__dict__'):
                logger.info(f"Input Attributes: {list(input_data.__dict__.keys())}")
                logger.info("Input Data:")
                logger.info(self._format_data(input_data.__dict__))
            else:
                logger.info("Input Data:")
                logger.info(self._format_data(input_data))
        
        logger.info("-" * 40)
    
    def log_agent_result(self, logger: logging.Logger, agent_name: str, result: Any):
        """Log the result of an agent's processing"""
        logger.info(f"📊 AGENT RESULT: {agent_name.upper()}")
        
        if hasattr(result, 'success'):
            logger.info(f"Success: {result.success}")
            if not result.success and hasattr(result, 'errors'):
                logger.error(f"Errors: {result.errors}")
            
            if hasattr(result, 'data') and result.data:
                logger.info(f"Result Data Type: {type(result.data).__name__}")
                if isinstance(result.data, dict):
                    logger.info(f"Result Data Keys: {list(result.data.keys())}")
                    logger.info("Result Data:")
                    logger.info(self._format_data(result.data))
                elif isinstance(result.data, (list, tuple)):
                    logger.info(f"Result Data: {len(result.data)} items")
                    if len(result.data) > 0:
                        logger.info("Sample items:")
                        sample_size = min(3, len(result.data))
                        for i in range(sample_size):
                            logger.info(f"  Item {i+1}: {self._format_data(result.data[i], max_length=200)}")
                else:
                    logger.info("Result Data:")
                    logger.info(self._format_data(result.data))
                
            if hasattr(result, 'metadata') and result.metadata:
                logger.info("Metadata:")
                logger.info(self._format_data(result.metadata))
        else:
            logger.info(f"Result Type: {type(result).__name__}")
            logger.info("Result:")
            logger.info(self._format_data(result, max_length=300))
        
        logger.info("-" * 40)
    
    def log_sql_query(self, logger: logging.Logger, query: str, parameters: Dict[str, Any] = None):
        """Log SQL query execution details"""
        logger.info("🔍 SQL QUERY EXECUTION")
        logger.info("Query:")
        logger.info(query)
        if parameters:
            logger.info("Parameters:")
            logger.info(self._format_data(parameters))
        logger.info("-" * 40)
    
    def log_database_result(self, logger: logging.Logger, row_count: int, execution_time: float, 
                           columns: list = None):
        """Log database query results"""
        logger.info("💾 DATABASE RESULT")
        logger.info(f"Rows returned: {row_count}")
        logger.info(f"Execution time: {execution_time:.3f} seconds")
        if columns:
            logger.info(f"Columns: {columns}")
        logger.info("-" * 40)
    
    def log_visualization_creation(self, logger: logging.Logger, viz_types: list, file_paths: list = None):
        """Log visualization creation details"""
        logger.info("📊 VISUALIZATIONS CREATED")
        logger.info(f"Types: {viz_types}")
        if file_paths:
            logger.info("Files:")
            for path in file_paths:
                logger.info(f"  {path}")
        logger.info("-" * 40)
    
    def log_error(self, logger: logging.Logger, agent_name: str, error: Exception, context: str = None):
        """Log error with context"""
        logger.error(f"❌ ERROR in {agent_name.upper()}")
        if context:
            logger.error(f"Context: {context}")
        logger.error(f"Error Type: {type(error).__name__}")
        logger.error(f"Error Message: {str(error)}")
        logger.error("-" * 40)
    
    def close_query_logger(self, logger: logging.Logger):
        """Close a query logger and clean up resources"""
        logger.info("="*80)
        logger.info(f"QUERY PROCESSING COMPLETED - {datetime.now().isoformat()}")
        logger.info("="*80)
        
        # Close all handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # Remove from cache by finding the matching logger
        cache_key_to_remove = None
        for key, cached_logger in self._loggers_cache.items():
            if cached_logger is logger:
                cache_key_to_remove = key
                break
        
        if cache_key_to_remove:
            del self._loggers_cache[cache_key_to_remove]
    
    def cleanup_old_logs(self, max_age_hours: int = 168):  # 7 days default
        """Clean up old log files"""
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        for log_file in self.log_dir.glob("*.log"):
            try:
                file_age = current_time - log_file.stat().st_mtime
                if file_age > max_age_seconds:
                    log_file.unlink()
                    cleaned_count += 1
            except Exception:
                pass  # Ignore cleanup errors
        
        return cleaned_count
    
    def list_query_logs(self, limit: int = 20) -> list:
        """List recent query log files"""
        log_files = list(self.log_dir.glob("*.log"))
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        result = []
        for log_file in log_files[:limit]:
            try:
                stat = log_file.stat()
                result.append({
                    'filename': log_file.name,
                    'path': str(log_file),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception:
                continue
        
        return result
    
    def get_query_logger_wrapper(self, query: str, session_id: Optional[str] = None) -> 'QueryLoggerWrapper':
        """Get a wrapper that combines QueryLogger methods with standard logging methods"""
        logger = self.get_query_logger(query, session_id)
        return QueryLoggerWrapper(self, logger)


class QueryLoggerWrapper:
    """Wrapper class that provides both QueryLogger methods and standard logging methods"""
    
    def __init__(self, query_logger_manager: QueryLogger, logger: logging.Logger):
        self.query_logger_manager = query_logger_manager
        self.logger = logger
    
    # Direct method calls that agents expect
    def log_agent_start(self, agent_name: str, input_data: Any = None):
        """Log agent start using the QueryLogger manager"""
        self.query_logger_manager.log_agent_start(self.logger, agent_name, input_data)
    
    def log_result(self, agent_name: str, result: Any):
        """Log agent result using the QueryLogger manager"""
        self.query_logger_manager.log_agent_result(self.logger, agent_name, result)
    
    def log_error(self, agent_name: str, error: str, context: str = ""):
        """Log error using the QueryLogger manager"""
        if isinstance(error, str):
            # Create a simple exception from string
            error_obj = Exception(error)
        else:
            error_obj = error
        self.query_logger_manager.log_error(self.logger, agent_name, error_obj, context)
    
    def log_sql_query(self, sql_query: str, parameters: dict = None):
        """Log SQL query using the QueryLogger manager"""
        self.query_logger_manager.log_sql_query(self.logger, sql_query, parameters)
    
    def log_database_result(self, row_count: int, execution_time: float, columns: list):
        """Log database result using the QueryLogger manager"""
        self.query_logger_manager.log_database_result(self.logger, row_count, execution_time, columns)
    
    def cleanup(self):
        """Close the logger"""
        self.query_logger_manager.close_query_logger(self.logger)
    
    # Standard logging methods
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)


# Global instance
_global_query_logger = None

def get_query_logger(log_dir: str = "logs/queries") -> QueryLogger:
    """Get or create global query logger instance"""
    global _global_query_logger
    if _global_query_logger is None:
        _global_query_logger = QueryLogger(log_dir)
    return _global_query_logger


def create_query_logger(query: str, session_id: Optional[str] = None, 
                       log_dir: str = "logs/queries") -> logging.Logger:
    """Convenience function to create a query-specific logger"""
    query_logger_manager = get_query_logger(log_dir)
    return query_logger_manager.get_query_logger(query, session_id)