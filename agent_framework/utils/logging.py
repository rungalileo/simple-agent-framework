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
from abc import ABC, abstractmethod
from agent_framework.utils.hooks import ToolHooks, ToolSelectionHooks


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

class AgentLogger(ABC):
    """Abstract base class for agent logging"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._tool_hooks = None
        self._tool_selection_hooks = None

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log an informational message"""
        pass
        
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message"""
        pass
        
    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log an error message"""
        pass
        
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message"""
        pass

    @abstractmethod
    def _write_log(self, log_entry: Dict[str, Any]) -> None:
        """Write a log entry"""
        pass

    @abstractmethod
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Sanitize an object for JSON serialization"""
        pass

    @abstractmethod
    async def on_agent_planning(self, planning_prompt: str) -> None:
        """Log the agent planning prompt"""
        pass

    @abstractmethod
    def on_agent_start(self, initial_task: str) -> None:
        """Log the agent execution prompt"""
        pass

    @abstractmethod
    async def on_agent_done(self, result: str, message_history: List[Dict[str, Any]]) -> None:
        """Log the agent completion"""
        pass

    def get_tool_hooks(self) -> ToolHooks:
        """Get tool hooks for this logger"""
        return self._tool_hooks

    def get_tool_selection_hooks(self) -> ToolSelectionHooks:
        """Get tool selection hooks for this logger"""
        return self._tool_selection_hooks
    
class ConsoleAgentLogger(AgentLogger):
    """Console implementation of agent logger"""
    
    def info(self, message: str, **kwargs) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"[timestamp]{timestamp}[/timestamp] [info]INFO[/info]: {message}")
        if kwargs:
            console.print(Panel(json.dumps(kwargs, indent=2), title="Additional Info"))
            
    def warning(self, message: str, **kwargs) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"[timestamp]{timestamp}[/timestamp] [warning]WARNING[/warning]: {message}")
        if kwargs:
            console.print(Panel(json.dumps(kwargs, indent=2), title="Additional Info"))
            
    def error(self, message: str, **kwargs) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"[timestamp]{timestamp}[/timestamp] [error]ERROR[/error]: {message}")
        if kwargs:
            console.print(Panel(json.dumps(kwargs, indent=2), title="Additional Info"))
            
    def debug(self, message: str, **kwargs) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"[timestamp]{timestamp}[/timestamp] [dim]DEBUG[/dim]: {message}")
        if kwargs:
            console.print(Panel(json.dumps(kwargs, indent=2), title="Additional Info"))

    def _write_log(self, log_entry: Dict[str, Any]) -> None:
        pass  # Console logger doesn't need to write to file

    def _sanitize_for_json(self, obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        else:
            return str(obj)

    async def on_agent_planning(self, planning_prompt: str) -> None:
        self.info(f"Planning: {planning_prompt}")

    def on_agent_start(self, initial_task: str) -> None:
        self.info(f"Starting task: {initial_task}")

    async def on_agent_done(self, result: str, message_history: List[Dict[str, Any]]) -> None:
        self.info(f"Task completed: {result}")
