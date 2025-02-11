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

class AsyncWorkflowWrapper:
    """Simple wrapper for Galileo workflow operations"""
    def __init__(self, workflow: AgentStep):
        self._workflow = workflow

    async def add_llm(self, **kwargs): return self._workflow.add_llm(**kwargs)
    async def add_tool(self, **kwargs): return self._workflow.add_tool(**kwargs)
    async def conclude(self, **kwargs): return self._workflow.conclude(**kwargs)


class EventQueue:
    """Manages ordered event processing for Galileo logging"""
    def __init__(self):
        self._events: List[Event] = []
        self._counter = 0
        self._start_time = datetime.now()
        self._lock = asyncio.Lock()
        self._processing = False
        self._workflow = None

    def set_workflow(self, workflow: AsyncWorkflowWrapper):
        """Set the current workflow for logging"""
        self._workflow = workflow

    async def add(self, event: Event):
        """Add an event with ordering metadata and process queue"""
        # Add ordering metadata
        async with self._lock:
            self._counter += 1
            timestamp = (datetime.now() - self._start_time).total_seconds()
            event.metadata.update({
                "sequence": str(self._counter),
                "timestamp": str(timestamp),
                "type": "event"
            })
            self._events.append(event)
        
        # Process queue outside of lock
        await self._process_queue()

    async def _process_queue(self):
        """Process events in order"""
        if not self._workflow or not self._events:
            return

        if self._processing:
            return
            
        async with self._lock:
            self._processing = True
            try:
                # Sort events by sequence number
                self._events.sort(key=lambda e: int(e.metadata["sequence"]))
                
                # Process each event
                while self._events:
                    event = self._events[0]
                    try:
                        if event.type == 'llm':
                            await self._workflow.add_llm(
                                name=event.name,
                                input=event.input,
                                output=event.output,
                                tools=event.tools,
                                model=event.model,
                                metadata=event.metadata
                            )
                        else:  # tool
                            await self._workflow.add_tool(
                                name=event.name,
                                input=event.input,
                                output=event.output,
                                metadata=event.metadata
                            )
                        self._events.pop(0)
                    except Exception:
                        self._events.pop(0)
                    await asyncio.sleep(0.1)  # Ensure Galileo processes in order
            finally:
                self._processing = False


class GalileoLogger:
    """Core logging functionality for Galileo"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.queue = EventQueue()
        self._workflow = None

    @property
    def workflow(self) -> AsyncWorkflowWrapper:
        if not self._workflow:
            raise RuntimeError("Workflow not initialized")
        return self._workflow

    @workflow.setter
    def workflow(self, value: AsyncWorkflowWrapper):
        self._workflow = value
        self.queue.set_workflow(value)

    async def log_llm(self, name: str, input: Any, output: Any = "", **kwargs):
        """Log an LLM event"""
        metadata = {"agent_id": self.agent_id, **kwargs.get("metadata", {})}
        await self.queue.add(Event("llm", name, input, output, metadata, 
                                 kwargs.get("tools"), kwargs.get("model")))

    async def log_tool(self, name: str, input: Any, output: Any = "", **kwargs):
        """Log a tool event"""
        metadata = {"agent_id": self.agent_id, **kwargs.get("metadata", {})}
        await self.queue.add(Event("tool", name, input, output, metadata))
