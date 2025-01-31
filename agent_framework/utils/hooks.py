from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ToolContext:
    """Context passed to tool hooks containing execution history and metadata"""
    task: str = field(metadata={"description": "The current task being executed"})
    tool_name: str = field(metadata={"description": "Name of the tool being executed"})
    inputs: Dict[str, Any] = field(metadata={"description": "Input parameters for the tool"})
    previous_tools: List[str] = field(metadata={"description": "List of tools that were previously executed in this task"})
    previous_results: List[Dict[str, Any]] = field(metadata={"description": "Results from previous tool executions"})
    previous_errors: List[str] = field(metadata={"description": "Errors from previous tool executions"})
    message_history: List[Dict[str, Any]] = field(metadata={"description": "Complete history of messages and tool executions"})
    agent_id: str = field(metadata={"description": "ID of the agent executing the tool"})
    task_id: str = field(metadata={"description": "ID of the current task"})
    start_time: datetime = field(metadata={"description": "When the task started"})
    metadata: Dict[str, Any] = field(metadata={"description": "Additional metadata from agent configuration"})

class ToolHooks:
    """Hooks for tool execution lifecycle"""
    
    async def before_execution(self, context: ToolContext) -> None:
        """Called before tool execution with full context"""
        pass
        
    async def after_execution(self, context: ToolContext, result: Any, error: Optional[Exception] = None) -> None:
        """Called after tool execution with the result or error"""
        pass

class ToolSelectionHooks:
    """Hooks for tool selection lifecycle"""
    
    async def after_selection(
        self,
        context: ToolContext,
        selected_tool: str,
        confidence: float,
        reasoning: List[str]
    ) -> None:
        """Called after tool selection with selection details"""
        pass 