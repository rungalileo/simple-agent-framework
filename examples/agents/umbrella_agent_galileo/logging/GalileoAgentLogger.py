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
                        "inputs": msg.get("inputs", {}),
                        "result": msg.get("result", {}),
                        "reasoning": msg.get("reasoning", "")
                    })
                })
            else:
                # For other messages, keep the role and content if available
                formatted.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", json.dumps(msg))
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

class LogEvent:
    """Represents a single logging event"""
    def __init__(self, event_type: str, name: str, input: Any, output: Any, metadata: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None, model: Optional[str] = None):
        self.event_type = event_type  # 'llm' or 'tool'
        self.name = name
        self.input = input
        self.output = output
        self.metadata = metadata
        self.tools = tools
        self.model = model

class QueuedLogger:
    """Singleton class that maintains a queue of logging events"""
    _instance = None
    _lock = asyncio.Lock()
    _events: List[LogEvent] = []
    _is_processing = False
    _current_workflow = None
    _processing_lock = asyncio.Lock()
    _event_counter = 0  # Global event counter
    _start_time = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._start_time = datetime.now()
        return cls._instance

    @classmethod
    async def get_next_order(cls) -> Dict[str, Any]:
        """Get next event order metadata"""
        async with cls._lock:
            cls._event_counter += 1
            # Calculate timestamp with microsecond precision for strict ordering
            timestamp = (datetime.now() - cls._start_time).total_seconds()
            return {
                "sequence": str(cls._event_counter),
                "timestamp": str(timestamp),
                "type": "event"
            }

    @classmethod
    def set_workflow(cls, workflow: 'AsyncWorkflowWrapper'):
        """Set the current workflow for logging"""
        print(f"[QueuedLogger] Setting workflow: {id(workflow)}")
        cls._current_workflow = workflow

    @classmethod
    async def queue_event(cls, event: LogEvent):
        """Add an event to the queue and process it"""
        if not event.metadata:
            event.metadata = {}
            
        # Merge order metadata with existing metadata
        order_metadata = await cls.get_next_order()
        event.metadata.update(order_metadata)
            
        print(f"[QueuedLogger] Queueing event: type={event.event_type}, name={event.name}, sequence={event.metadata.get('sequence')}, timestamp={event.metadata.get('timestamp')}")
        async with cls._lock:
            cls._events.append(event)
            print(f"[QueuedLogger] Current queue length: {len(cls._events)}")
            if not cls._is_processing:
                print("[QueuedLogger] Starting event processing")
                await cls._process_events()
            else:
                print("[QueuedLogger] Event processing already in progress")

    @classmethod
    async def _process_events(cls):
        """Process all events in the queue in order"""
        if cls._current_workflow is None:
            print("[QueuedLogger] Warning: No workflow set, events will be dropped")
            return

        async with cls._processing_lock:
            if cls._is_processing:
                print("[QueuedLogger] Already processing events, skipping")
                return
                
            cls._is_processing = True
            print(f"[QueuedLogger] Starting to process {len(cls._events)} events")
            try:
                # Sort events by sequence number before processing
                cls._events.sort(key=lambda e: int(e.metadata.get('sequence', '0')))
                print(f"[QueuedLogger] Events sorted by sequence: {[(e.name, e.metadata.get('sequence')) for e in cls._events]}")
                
                while cls._events:
                    event = cls._events[0]
                    print(f"[QueuedLogger] Processing event: type={event.event_type}, name={event.name}, sequence={event.metadata.get('sequence')}, timestamp={event.metadata.get('timestamp')}")
                    try:
                        if event.event_type == 'llm':
                            event.output = ensure_valid_io(event.output)
                            event.input = ensure_valid_io(event.input)
                            print(f"[QueuedLogger] Adding LLM step: {event.name}")
                            await cls._current_workflow.add_llm(
                                name=event.name,
                                input=event.input,
                                output=event.output,
                                tools=event.tools,
                                model=event.model or "gpt-4o",
                                metadata=event.metadata
                            )
                            # Wait a small amount to ensure Galileo processes in order
                            await asyncio.sleep(0.1)
                        elif event.event_type == 'tool':
                            event.output = ensure_valid_io(event.output)
                            event.input = ensure_valid_io(event.input)
                            print(f"[QueuedLogger] Adding Tool step: {event.name}")
                            await cls._current_workflow.add_tool(
                                name=event.name,
                                input=event.input,
                                output=event.output,
                                metadata=event.metadata
                            )
                            # Wait a small amount to ensure Galileo processes in order
                            await asyncio.sleep(0.1)
                        print(f"[QueuedLogger] Successfully processed event: {event.name}")
                        cls._events.pop(0)
                    except Exception as e:
                        print(f"[QueuedLogger] Error processing event {event.name}: {e}")
                        cls._events.pop(0)
            finally:
                cls._is_processing = False
                print("[QueuedLogger] Finished processing events")

class AsyncWorkflowWrapper:
    """Async wrapper for Galileo workflow operations"""
    
    def __init__(self, workflow: AgentStep):
        self._workflow = workflow

    async def add_llm(self, **kwargs):
        """Execute add_llm operation directly"""
        return self._workflow.add_llm(**kwargs)

    async def add_tool(self, **kwargs):
        """Execute add_tool operation directly"""
        return self._workflow.add_tool(**kwargs)

    async def conclude(self, **kwargs):
        """Execute conclude operation directly"""
        return self._workflow.conclude(**kwargs)

class AsyncObserveWrapper:
    """Async wrapper for Galileo ObserveWorkflows"""
    
    def __init__(self, observe_logger: ObserveWorkflows):
        self._observe_logger = observe_logger
        self._logger = QueuedLogger()

    async def add_agent_workflow(self, **kwargs):
        """Create a new workflow and initialize it"""
        # Create workflow first
        workflow = self._observe_logger.add_agent_workflow(**kwargs)
        wrapped = AsyncWorkflowWrapper(workflow)
        
        # Set it as current workflow before logging anything
        self._logger.set_workflow(wrapped)
        
        # Now we can safely log events
        metadata = kwargs.get('metadata', {})
        await self._logger.queue_event(LogEvent(
            event_type='llm',
            name='umbrella_agent',
            input=kwargs['input'],
            output="",
            metadata=metadata
        ))
        
        return wrapped

    async def upload_workflows(self):
        """Actually upload the workflows"""
        # Process any remaining events before uploading
        await self._logger._process_events()
        # Then upload
        self._observe_logger.upload_workflows()

class GalileoToolHooks:
    def __init__(self, logger: 'GalileoAgentLogger'):
        self.logger = logger
        self._logger = QueuedLogger()

    @property
    def workflow(self) -> AgentStep:
        return self.logger.workflow

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
            await self._logger.queue_event(LogEvent(
                event_type='tool',
                name=context.tool_name,
                input=context.inputs,
                output=result,
                metadata={"agent_id": context.agent_id, "type": "execution"}
            ))

class GalileoToolSelectionHooks:
    def __init__(self, logger: 'GalileoAgentLogger'):
        self.logger = logger
        self._logger = QueuedLogger()

    @property
    def workflow(self) -> AgentStep:
        return self.logger.workflow

    async def after_selection(
        self,
        context: ToolContext,
        selected_tool: str,
        confidence: float,
        reasoning: List[str]
    ) -> None:
        """Log tool selection"""
        await self._logger.queue_event(LogEvent(
            event_type='llm',
            name=f"{selected_tool}_selection",
            input=format_messages(context.message_history) if context.message_history else [],
            output={
                "selected_tool": selected_tool,
                "confidence": confidence,
                "reasoning": reasoning
            },
            tools=context.available_tools,
            model="gpt-4o",
            metadata={"agent_id": context.agent_id, "type": "selection"}
        ))

class GalileoAgentLogger(AgentLogger):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.observe_logger = AsyncObserveWrapper(observe_logger)
        self._workflow = None
        self._logger = QueuedLogger()
        print(f"[GalileoAgentLogger] Initialized with agent_id: {agent_id}")

    @property
    def workflow(self) -> AsyncWorkflowWrapper:
        if self._workflow is None:
            raise RuntimeError("Workflow not initialized. Must call on_agent_planning first.")
        return self._workflow

    @workflow.setter
    def workflow(self, value: AsyncWorkflowWrapper):
        self._workflow = value
        self._logger.set_workflow(value)

    async def add_llm_step(self, **kwargs) -> None:
        """Queue an LLM event"""
        print(f"[GalileoAgentLogger] Adding LLM step: name={kwargs.get('name')}")
        await self._logger.queue_event(LogEvent(event_type='llm', **kwargs))

    async def add_tool_step(self, **kwargs) -> None:
        """Queue a tool event"""
        print(f"[GalileoAgentLogger] Adding Tool step: name={kwargs.get('name')}")
        await self._logger.queue_event(LogEvent(event_type='tool', **kwargs))

    async def add_workflow(
        self,
        input: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncWorkflowWrapper:
        """Async wrapper for observe_logger.add_agent_workflow"""
        return await self.observe_logger.add_agent_workflow(
            input=input,
            name=name,
            metadata=metadata or {}
        )

    def info(self, message: str, **kwargs) -> None:
        print(f"INFO: {message}") # Don't change this

    def warning(self, message: str, **kwargs) -> None:
        print(f"WARNING: {message}") # Don't change this

    def error(self, message: str, **kwargs) -> None:
        print(f"ERROR: {message}") # Don't change this

    def debug(self, message: str, **kwargs) -> None:
        print(f"DEBUG: {message}") # Don't change this

    def _write_log(self, log_entry: Dict[str, Any]) -> None:
        # Not needed as we use observe_logger directly
        pass

    def _sanitize_for_json(self, obj: Any) -> Any:
        # Not needed as observe_logger handles serialization
        pass

    def get_logger(self) -> ObserveWorkflows:
        return self.observe_logger

    def get_tool_hooks(self) -> ToolHooks:
        """Get tool hooks for this logger"""
        return GalileoToolHooks(self)
        
    def get_tool_selection_hooks(self) -> ToolSelectionHooks:
        """Get tool selection hooks for this logger"""
        return GalileoToolSelectionHooks(self)
    
    async def on_agent_planning(self, planning_prompt: str) -> None:
        """Log agent planning event"""
        print("[GalileoAgentLogger] Starting agent planning")
        # Initialize workflow first
        workflow = await self.add_workflow(
            input=planning_prompt,
            name="umbrella_agent",
            metadata={"agent_id": self.agent_id}
        )
        
        print("[GalileoAgentLogger] Setting workflow")
        self.workflow = workflow
        
        print("[GalileoAgentLogger] Adding planning step")
        await self.add_llm_step(
            name="agent_planning",
            input=planning_prompt,
            output="",
            metadata={"agent_id": self.agent_id, "type": "planning"}
        )

    def on_agent_start(self, initial_task: str) -> None:
        """Log the agent execution prompt"""
        print(f"Initial task: {initial_task}")

    async def on_agent_done(self, result: Any, message_history: Optional[List[Any]] = None) -> None:
        """Log agent completion event"""
        print("[GalileoAgentLogger] Starting agent completion")
        # First add the final result
        await self.add_llm_step(
            name="final_result",
            input=ensure_valid_io(format_messages(message_history) if message_history else []),
            output=ensure_valid_io(result),
            model="gpt-4o",
            metadata={"agent_id": self.agent_id, "type": "final_result"}
        )
        
        print("[GalileoAgentLogger] Processing remaining events")
        await self._logger._process_events()
        
        print("[GalileoAgentLogger] Concluding workflow")
        await self.workflow.conclude(output={"result": result})
        
        print("[GalileoAgentLogger] Uploading workflows")
        await self.observe_logger.upload_workflows()
        print("[GalileoAgentLogger] Agent completion finished")

    async def on_tool_selection(self, tool_name: str, tool_input: Any) -> None:
        """Log tool selection event"""
        await self.add_tool_step(
            name=tool_name,
            input=tool_input,
            output="",
            metadata={"agent_id": self.agent_id, "type": "selection"}
        )

    async def on_tool_execution(self, tool_name: str, tool_input: Any, tool_output: Any) -> None:
        """Log tool execution event"""
        await self.add_tool_step(
            name=tool_name,
            input=tool_input,
            output=tool_output,
            metadata={"agent_id": self.agent_id, "type": "execution"}
        )
