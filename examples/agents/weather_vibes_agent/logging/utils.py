from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
from agent_framework.utils.validation import ensure_valid_io
from galileo_observe import AgentStep

class Event:
    """Represents a single logging event with ordering metadata"""
    def __init__(self, type: str, name: str, input: Any, output: Any, metadata: Dict[str, Any], 
                 tools: Optional[List[Dict[str, Any]]] = None, model: Optional[str] = None):
        self.type = type  # 'llm' or 'tool'
        self.name = name
        self.input = ensure_valid_io(input)
        self.output = ensure_valid_io(output)
        self.metadata = metadata or {}
        self.tools = tools
        self.model = model or "gpt-4o"

class EventQueue:
    """Queue for logging events"""
    def __init__(self):
        self.events = []
        
    def add_event(self, event: Event):
        self.events.append(event)
        
    def get_events(self):
        return self.events

class GalileoLogger:
    """Logger for Galileo"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.event_queue = EventQueue()
        self.workflow = None
        
    async def log_llm(self, name: str, input: Any, output: Any, metadata: Dict[str, Any] = None, 
                      tools: Optional[List[Dict[str, Any]]] = None, model: Optional[str] = None):
        """Log an LLM event"""
        if self.workflow:
            await self.workflow.add_llm(
                name=name,
                input=input,
                output=output,
                metadata=metadata or {},
                tools=tools,
                model=model or "gpt-4o"
            )
        
    async def log_tool(self, name: str, input: Any, output: Any, metadata: Dict[str, Any] = None):
        """Log a tool event"""
        if self.workflow:
            await self.workflow.add_tool(
                name=name,
                input=input,
                output=output,
                metadata=metadata or {}
            )

class AsyncWorkflowWrapper:
    """Simple wrapper for Galileo workflow operations"""
    def __init__(self, workflow: AgentStep):
        self._workflow = workflow

    async def add_llm(self, **kwargs): return self._workflow.add_llm(**kwargs)
    async def add_tool(self, **kwargs): return self._workflow.add_tool(**kwargs)
    async def conclude(self, **kwargs): return self._workflow.conclude(**kwargs) 