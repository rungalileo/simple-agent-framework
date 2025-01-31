from typing import Any, Dict, List, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme
from rich.box import ROUNDED
import json

from .hooks import ToolContext, ToolHooks, ToolSelectionHooks
from ..llm.models import LLMMessage

# Create a custom theme for our logger
theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "timestamp": "dim cyan",
    "tool": "magenta",
    "reasoning": "blue",
    "confidence": "yellow"
})

console = Console(theme=theme)

class AgentLogger:
    """Logger for recording agent activity"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
    def info(self, message: str, **kwargs) -> None:
        """Log an informational message"""
        timestamp = datetime.utcnow().isoformat()
        
        # Convert LLMMessages to dict for serialization
        processed_kwargs = {}
        for key, value in kwargs.items():
            if key == "prompt_messages" and isinstance(value, list):
                processed_kwargs[key] = [
                    {"role": msg.role, "content": msg.content}
                    for msg in value
                ]
            else:
                processed_kwargs[key] = value
        
        log_entry = {
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "level": "INFO",
            "message": message,
            **processed_kwargs
        }
        self._write_log(log_entry)
        
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "level": "WARNING",
            "message": message,
            **kwargs
        }
        self._write_log(log_entry)
        
    def error(self, message: str, **kwargs) -> None:
        """Log an error message"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "level": "ERROR",
            "message": message,
            **kwargs
        }
        self._write_log(log_entry)
        
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "level": "DEBUG",
            "message": message,
            **kwargs
        }
        self._write_log(log_entry)

    def _write_log(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to storage/file"""
        # For now, just print the JSON log entry
        # In a real implementation, this would write to a file or database
        try:
            print(f"LOG: {json.dumps(log_entry)}")
        except TypeError as e:
            # If serialization fails, try to convert problematic values to strings
            sanitized_entry = self._sanitize_for_json(log_entry)
            print(f"LOG: {json.dumps(sanitized_entry)}")
            
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively sanitize an object for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert any other type to string representation
            return str(obj)

class LoggingToolHooks(ToolHooks):
    """Tool hooks that log execution details"""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        
    async def before_execution(self, context: ToolContext) -> None:
        """Log before tool execution"""
        self.logger.info(
            f"Executing tool: {context.tool_name}",
            inputs=context.inputs,
            task_id=context.task_id
        )
        
    async def after_execution(
        self,
        context: ToolContext,
        result: Any,
        error: Optional[Exception] = None
    ) -> None:
        """Log after tool execution"""
        if error:
            self.logger.error(
                f"Tool execution failed: {context.tool_name}",
                error=str(error),
                task_id=context.task_id
            )
        else:
            self.logger.info(
                f"Tool execution completed: {context.tool_name}",
                result=result,
                task_id=context.task_id
            )

class LoggingToolSelectionHooks(ToolSelectionHooks):
    """Tool selection hooks that log selection details"""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        
    async def after_selection(
        self,
        context: ToolContext,
        selected_tool: str,
        confidence: float,
        reasoning: List[str]
    ) -> None:
        """Log tool selection details"""
        self.logger.info(
            f"Selected tool: {selected_tool}",
            confidence=confidence,
            reasoning=reasoning,
            task_id=context.task_id
        )
