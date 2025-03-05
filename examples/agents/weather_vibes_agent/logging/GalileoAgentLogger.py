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

    def on_agent_start(self, initial_task: str): pass
    def _write_log(self, log_entry: Dict[str, Any]): pass
    def _sanitize_for_json(self, obj: Any): pass 