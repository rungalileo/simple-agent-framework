from galileo_observe import ObserveWorkflows, AgentStep
from agent_framework.utils.logging import AgentLogger
from agent_framework.utils.tool_hooks import create_tool_hooks, create_tool_selection_hooks
from agent_framework.utils.hooks import ToolHooks, ToolSelectionHooks
from typing import Any, List, Optional, Dict, Sequence, Union
from agent_framework.utils.tool_registry import ToolRegistry
from agent_framework.models import ToolContext
from agent_framework.llm.models import LLMMessage
import json
import dotenv

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

class GalileoAgentLogger(AgentLogger):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.observe_logger = observe_logger
        self.wf = observe_logger.add_workflow(
            input={"agent_id": agent_id},
            name="umbrella_agent",
            metadata={"agent_id": agent_id}
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

    def get_workflow(self) -> AgentStep:
        return self.wf
    
    async def upload_workflows(self) -> None:
        """Upload all workflows to Galileo"""
        print("Uploading workflows to Galileo...")
        self.observe_logger.upload_workflows()
        print("Workflows uploaded successfully")

    def get_tool_hooks(self) -> ToolHooks:
        """Get tool hooks for this logger"""
        return GalileoToolHooks(self)
        
    def get_tool_selection_hooks(self) -> ToolSelectionHooks:
        """Get tool selection hooks for this logger"""
        return GalileoToolSelectionHooks(self)
    
    def on_agent_planning(self, planning_prompt: str) -> None:
        """Log the agent planning prompt"""
        print(f"Planning prompt: {planning_prompt}")

    def on_agent_start(self, initial_task: str) -> None:
        """Log the agent execution prompt"""
        print(f"Initial task: {initial_task}")

class GalileoToolHooks(ToolHooks):
    def __init__(self, logger: GalileoAgentLogger):
        self.logger = logger
        self.wf = logger.get_workflow()

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
            print(context.message_history)
            # Add error result to workflow
            # self.wf.add_llm(
            #     input=format_messages(context.message_history) if context.message_history else [],
            #     output=json.dumps({"error": str(error)}),
            #     model="gpt-4-mini",
            #     metadata={
            #         "agent_id": context.agent_id,
            #         "status": "error"
            #     }
            # )
        else:
            self.logger.info(
                f"Tool execution completed: {context.tool_name}",
                result=result,
                task_id=context.task_id
            )
            # Add tool execution to workflow
            print(f"I executed {context.tool_name} with inputs {context.inputs} and result {result}")
            self.wf.add_tool(
                name=context.tool_name,
                input=ensure_valid_io(context.inputs),
                output=ensure_valid_io(result),
                metadata={"agent_id": context.agent_id}
            )

class GalileoToolSelectionHooks(ToolSelectionHooks):
    def __init__(self, logger: GalileoAgentLogger):
        self.logger = logger
        self.wf = logger.get_workflow()

    async def after_selection(
        self,
        context: ToolContext,
        selected_tool: str,
        confidence: float,
        reasoning: List[str]
    ) -> None:
        """Log tool selection"""
        print(f"I selected {selected_tool} with confidence {confidence} and reasoning {reasoning}")
        self.wf.add_llm(
            input=format_messages(context.message_history) if context.message_history else [],
            output=json.dumps({
                "selected_tool": selected_tool,
                "confidence": confidence,
                "reasoning": reasoning
            }),
            tools=context.available_tools,
            model="gpt-4-mini",
            metadata={"agent_id": context.agent_id}
        )
