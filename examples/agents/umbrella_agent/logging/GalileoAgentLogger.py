from __future__ import annotations
from galileo_observe import ObserveWorkflows, AgentStep
from agent_framework.utils.logging import AgentLogger
from agent_framework.utils.hooks import ToolHooks, ToolSelectionHooks
from agent_framework.utils.validation import ensure_valid_io
from typing import Any, List, Optional, Dict, Sequence, Union, TYPE_CHECKING
from agent_framework.models import ToolContext
from agent_framework.llm.models import LLMMessage
import dotenv
from .utils import Event, EventQueue, GalileoLogger, AsyncWorkflowWrapper
if TYPE_CHECKING:
    from .GalileoAgentLogger import AsyncWorkflowWrapper

dotenv.load_dotenv()

class GalileoAgentLogger(AgentLogger):
    """Main logger interface for the agent"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.logger = GalileoLogger(agent_id)
        self.observe_logger = ObserveWorkflows(project_name=f"observe-{agent_id}")
        print(f"GalileoAgentLogger initialized for agent {agent_id}")

    async def on_agent_planning(self, planning_prompt: str) -> None:
        # Initialize workflow
        workflow = AsyncWorkflowWrapper(
            self.observe_logger.add_agent_workflow(
                input=planning_prompt,
                name=f"{self.agent_id}_planning",
                metadata={"agent_id": self.agent_id}
            )
        )
        self.logger.workflow = workflow

    async def on_agent_done(self, result: Any, message_history: Optional[List[Any]] = None) -> None:
        # Log final result
        await self.logger.log_llm(
            name="final_result",
            input=message_history or [],
            output=result,
            metadata={"type": "final_result"}
        )
        
        # Conclude and upload
        await self.logger.workflow.conclude(output={"result": result})
        self.observe_logger.upload_workflows()

    def get_tool_hooks(self) -> ToolHooks:
        """Create tool execution hooks"""
        logger = self.logger
        
        class Hooks(ToolHooks):
            async def before_execution(self, context: ToolContext) -> None:
                pass  # No logging needed for before execution

            async def after_execution(self, context: ToolContext, result: Any, 
                                   error: Optional[Exception] = None) -> None:
                if not error:
                    await logger.log_tool(
                        name=context.tool_name,
                        input=context.inputs,
                        output=result,
                        metadata={"type": "execution"}
                    )
        
        return Hooks()

    def get_tool_selection_hooks(self) -> ToolSelectionHooks:
        """Create tool selection hooks"""
        logger = self.logger
        
        class Hooks(ToolSelectionHooks):
            async def after_selection(self, context: ToolContext, selected_tool: str,
                                   confidence: float, reasoning: List[str]) -> None:
                # Always include complete context in the input
                input_data = {
                    "task": context.task,
                    "message_history": context.message_history or [],
                    "available_tools": context.available_tools,
                    "previous_tools": context.previous_tools,
                    "previous_results": context.previous_results
                }
                
                # Include plan if available
                if context.plan:
                    input_data["plan"] = {
                        "input_analysis": context.plan.input_analysis,
                        "available_tools": context.plan.available_tools,
                        "tool_capabilities": context.plan.tool_capabilities,
                        "execution_plan": context.plan.execution_plan,
                        "requirements_coverage": context.plan.requirements_coverage,
                        "chain_of_thought": context.plan.chain_of_thought
                    }

                await logger.log_llm(
                    name=f"{selected_tool}_selection",
                    input=input_data,
                    output={
                        "selected_tool": selected_tool,
                        "confidence": confidence,
                        "reasoning": reasoning
                    },
                    tools=context.available_tools,
                    metadata={"type": "selection"}
                )
        
        return Hooks()

    # Required but unused methods - keep minimal
    def info(self, message: str, **kwargs): pass
    def warning(self, message: str, **kwargs): pass
    def error(self, message: str, **kwargs): pass
    def debug(self, message: str, **kwargs): pass
    def on_agent_start(self, initial_task: str): pass
    def _write_log(self, log_entry: Dict[str, Any]): pass
    def _sanitize_for_json(self, obj: Any): pass
