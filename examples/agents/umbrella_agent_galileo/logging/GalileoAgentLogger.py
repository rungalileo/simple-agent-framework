from __future__ import annotations

from galileo_observe import ObserveWorkflows, AgentStep
from agent_framework.utils.logging import AgentLogger
from agent_framework.utils.tool_hooks import create_tool_hooks, create_tool_selection_hooks
from agent_framework.utils.hooks import ToolHooks, ToolSelectionHooks
from typing import Any, List, Optional, Dict, Sequence, Union, TYPE_CHECKING
from agent_framework.utils.tool_registry import ToolRegistry
from agent_framework.models import ToolContext
from agent_framework.llm.models import LLMMessage
import json
import dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio
from datetime import datetime

if TYPE_CHECKING:
    from .GalileoAgentLogger import AsyncWorkflowWrapper

dotenv.load_dotenv()

observe_logger = ObserveWorkflows(project_name="observe-umbrella-agent")

def format_messages(messages: Sequence[Union[LLMMessage, Dict[str, Any]]]) -> List[Dict[str, str]]:
    """Format messages into a list of role/content dictionaries.
    Handles both LLMMessage objects and pre-formatted dictionaries.
    """
    if not messages:
        return []
        
    def format_value(v: Any) -> Any:
        """Format a value for JSON serialization"""
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, dict):
            return {k: format_value(v) for k, v in v.items()}
        if isinstance(v, list):
            return [format_value(x) for x in v]
        return v

    formatted = []
    for msg in messages:
        if isinstance(msg, LLMMessage):
            formatted.append({
                "role": msg.role,
                "content": msg.content
            })
        elif isinstance(msg, dict):
            # For tool messages, format them appropriately
            if msg.get("role") == "tool":
                formatted.append({
                    "role": "tool",
                    "content": json.dumps({
                        "tool_name": msg.get("tool_name", ""),
                        "inputs": format_value(msg.get("inputs", {})),
                        "result": format_value(msg.get("result", {})),
                        "reasoning": msg.get("reasoning", "")
                    })
                })
            else:
                # For other messages, keep the role and content if available
                formatted.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", json.dumps(format_value(msg)))
                })
    return formatted

def ensure_valid_io(data: Any) -> str:
    """Ensure data is in a valid format for Galileo Step IO by converting to JSON string"""
    if data is None:
        return "{}"
    if isinstance(data, str):
        return data
    if isinstance(data, (dict, list)):
        return json.dumps(data)
    if isinstance(data, LLMMessage):
        return json.dumps({"role": data.role, "content": data.content})
    return json.dumps({"content": str(data)})

class Event:
    """Represents a single logging event with ordering metadata"""
    def __init__(self, type: str, name: str, input: Any, output: Any, metadata: Dict[str, Any], 
                 tools: Optional[List[Dict[str, Any]]] = None, model: Optional[str] = None):
        self.type = type  # 'llm' or 'tool'
        self.name = name
        self.input = self._ensure_valid_io(input)
        self.output = self._ensure_valid_io(output)
        self.metadata = metadata or {}
        self.tools = tools
        self.model = model or "gpt-4o"

    @staticmethod
    def _ensure_valid_io(data: Any) -> str:
        """Ensure data is in a valid format for Galileo Step IO"""
        if data is None:
            return "{}"
        if isinstance(data, str):
            return data
        if isinstance(data, datetime):
            return json.dumps(data.isoformat())
        if isinstance(data, (dict, list)):
            # Handle nested structures that might contain datetime objects
            def format_value(v: Any) -> Any:
                if isinstance(v, datetime):
                    return v.isoformat()
                if isinstance(v, dict):
                    return {k: format_value(v) for k, v in v.items()}
                if isinstance(v, list):
                    return [format_value(x) for x in v]
                return v
            return json.dumps(format_value(data))
        if isinstance(data, LLMMessage):
            return json.dumps({"role": data.role, "content": data.content})
        return json.dumps({"content": str(data)})

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
        print(f"Setting workflow: {id(workflow)}")
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
            print(f"Queueing event: {event.name} (sequence: {event.metadata['sequence']})")
            self._events.append(event)
        
        # Process queue outside of lock
        await self._process_queue()

    async def _process_queue(self):
        """Process events in order"""
        if not self._workflow:
            print("No workflow set, skipping event processing")
            return

        if self._processing:
            print("Already processing events")
            return
            
        async with self._lock:
            if not self._events:
                return
                
            self._processing = True
            try:
                # Sort events by sequence number
                self._events.sort(key=lambda e: int(e.metadata["sequence"]))
                print(f"Processing {len(self._events)} events")
                
                # Process each event
                while self._events:
                    event = self._events[0]
                    try:
                        print(f"Processing event: {event.name}")
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
                        print(f"Successfully processed event: {event.name}")
                        self._events.pop(0)
                    except Exception as e:
                        print(f"Error processing event {event.name}: {e}")
                        self._events.pop(0)
                    await asyncio.sleep(0.1)  # Ensure Galileo processes in order
            finally:
                self._processing = False
                print("Finished processing events")

class AsyncWorkflowWrapper:
    """Simple wrapper for Galileo workflow operations"""
    def __init__(self, workflow: AgentStep):
        self._workflow = workflow

    async def add_llm(self, **kwargs): return self._workflow.add_llm(**kwargs)
    async def add_tool(self, **kwargs): return self._workflow.add_tool(**kwargs)
    async def conclude(self, **kwargs): return self._workflow.conclude(**kwargs)

class GalileoLogger:
    """Core logging functionality for Galileo"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.queue = EventQueue()
        self._workflow = None
        print(f"Initialized GalileoLogger with agent_id: {agent_id}")

    @property
    def workflow(self) -> AsyncWorkflowWrapper:
        if not self._workflow:
            raise RuntimeError("Workflow not initialized")
        return self._workflow

    @workflow.setter
    def workflow(self, value: AsyncWorkflowWrapper):
        print(f"Setting workflow in GalileoLogger")
        self._workflow = value
        self.queue.set_workflow(value)

    async def log_llm(self, name: str, input: Any, output: Any = "", **kwargs):
        """Log an LLM event"""
        print(f"Logging LLM event: {name}")
        metadata = {"agent_id": self.agent_id, **kwargs.get("metadata", {})}
        await self.queue.add(Event("llm", name, input, output, metadata, 
                                 kwargs.get("tools"), kwargs.get("model")))

    async def log_tool(self, name: str, input: Any, output: Any = "", **kwargs):
        """Log a tool event"""
        print(f"Logging tool event: {name}")
        metadata = {"agent_id": self.agent_id, **kwargs.get("metadata", {})}
        await self.queue.add(Event("tool", name, input, output, metadata))

class GalileoAgentLogger(AgentLogger):
    """Main logger interface for the agent"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        print(f"Initializing GalileoAgentLogger with agent_id: {agent_id}")
        self.logger = GalileoLogger(agent_id)
        self.observe_logger = ObserveWorkflows(project_name="observe-umbrella-agent")

    async def on_agent_planning(self, planning_prompt: str) -> None:
        print("Starting agent planning")
        # Initialize workflow
        workflow = AsyncWorkflowWrapper(
            self.observe_logger.add_agent_workflow(
                input=planning_prompt,
                name="umbrella_agent",
                metadata={"agent_id": self.agent_id}
            )
        )
        self.logger.workflow = workflow
        
        # Log planning step
        print("Adding planning step")
        await self.logger.log_llm(
            name="agent_planning",
            input=planning_prompt,
            metadata={"type": "planning"}
        )
        print("Planning step completed")

    async def on_agent_done(self, result: Any, message_history: Optional[List[Any]] = None) -> None:
        print("Starting agent completion")
        # Log final result
        await self.logger.log_llm(
            name="final_result",
            input=message_history or [],
            output=result,
            metadata={"type": "final_result"}
        )
        
        # Conclude and upload
        print("Concluding workflow")
        await self.logger.workflow.conclude(output={"result": result})
        print("Uploading workflows")
        self.observe_logger.upload_workflows()
        print("Agent completion finished")

    def get_tool_hooks(self) -> ToolHooks:
        """Create tool execution hooks"""
        logger = self.logger
        
        class Hooks(ToolHooks):
            async def before_execution(self, context: ToolContext) -> None:
                """Required by ToolHooks but we don't need to log anything here"""
                print(f"INFO: Executing tool: {context.tool_name}")

            async def after_execution(self, context: ToolContext, result: Any, 
                                   error: Optional[Exception] = None) -> None:
                if not error:
                    print(f"INFO: Tool execution completed: {context.tool_name}")
                    await logger.log_tool(
                        name=context.tool_name,
                        input=context.inputs,
                        output=result,
                        metadata={"type": "execution"}
                    )
                else:
                    print(f"ERROR: Tool execution failed: {context.tool_name} - {error}")
        
        return Hooks()

    def get_tool_selection_hooks(self) -> ToolSelectionHooks:
        """Create tool selection hooks"""
        logger = self.logger
        
        class Hooks(ToolSelectionHooks):
            async def after_selection(self, context: ToolContext, selected_tool: str,
                                   confidence: float, reasoning: List[str]) -> None:
                await logger.log_llm(
                    name=f"{selected_tool}_selection",
                    input=context.message_history or [],
                    output={
                        "selected_tool": selected_tool,
                        "confidence": confidence,
                        "reasoning": reasoning
                    },
                    tools=context.available_tools,
                    metadata={"type": "selection"}
                )
        
        return Hooks()

    # Required but unused methods
    def info(self, message: str, **kwargs): print(f"INFO: {message}")
    def warning(self, message: str, **kwargs): print(f"WARNING: {message}")
    def error(self, message: str, **kwargs): print(f"ERROR: {message}")
    def debug(self, message: str, **kwargs): print(f"DEBUG: {message}")
    def on_agent_start(self, initial_task: str): print(f"Initial task: {initial_task}")
    def _write_log(self, log_entry: Dict[str, Any]): pass
    def _sanitize_for_json(self, obj: Any): pass
