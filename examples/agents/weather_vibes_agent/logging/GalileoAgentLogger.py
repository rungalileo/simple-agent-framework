from __future__ import annotations
from galileo_observe import ObserveWorkflows, AgentStep
from agent_framework.utils.logging import AgentLogger
from agent_framework.utils.hooks import ToolHooks, ToolSelectionHooks
from agent_framework.utils.validation import ensure_valid_io
from typing import Any, List, Optional, Dict, Sequence, Union, TYPE_CHECKING
from agent_framework.models import ToolContext
from agent_framework.llm.models import LLMMessage
import dotenv
import os
from .utils import Event, EventQueue, GalileoLogger, AsyncWorkflowWrapper
if TYPE_CHECKING:
    from .GalileoAgentLogger import AsyncWorkflowWrapper

dotenv.load_dotenv()

class GalileoAgentLogger(AgentLogger):
    """Main logger interface for the agent"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.logger = GalileoLogger(agent_id)
        
        # Check if Galileo 1.0 environment variables are available
        self.galileo_enabled = os.environ.get('GALILEO_CONSOLE_URL') is not None
        
        if self.galileo_enabled:
            try:
                self.observe_logger = ObserveWorkflows(project_name=f"observe-{agent_id}")
                print(f"GalileoAgentLogger initialized for agent {agent_id} with Galileo Observe")
            except Exception as e:
                print(f"Failed to initialize Galileo Observe: {e}")
                self.galileo_enabled = False
                self.observe_logger = None
        else:
            print(f"GalileoAgentLogger initialized for agent {agent_id} without Galileo Observe (GALILEO_CONSOLE_URL not set)")
            self.observe_logger = None

    async def on_agent_planning(self, planning_prompt: str) -> None:
        # Initialize workflow if Galileo 1.0 is enabled
        if self.galileo_enabled and self.observe_logger:
            try:
                workflow = AsyncWorkflowWrapper(
                    self.observe_logger.add_agent_workflow(
                        input=planning_prompt,
                        name=f"{self.agent_id}_planning",
                        metadata={"agent_id": self.agent_id}
                    )
                )
                self.logger.workflow = workflow
            except Exception as e:
                print(f"Error in on_agent_planning: {e}")
                self.logger.workflow = None
        else:
            # Simple logging fallback
            print(f"PLANNING [{self.agent_id}]: Starting planning with prompt: {planning_prompt[:100]}...")
            self.logger.workflow = None

    async def on_agent_done(self, result: Any, message_history: Optional[List[Any]] = None) -> None:
        # Log final result
        if self.galileo_enabled and hasattr(self.logger, 'log_llm'):
            try:
                await self.logger.log_llm(
                    name="final_result",
                    input=message_history or [],
                    output=result,
                    metadata={"type": "final_result"}
                )
                
                # Conclude and upload if workflow exists
                if hasattr(self.logger, 'workflow') and self.logger.workflow:
                    await self.logger.workflow.conclude(output={"result": result})
                    if self.observe_logger:
                        self.observe_logger.upload_workflows()
            except Exception as e:
                print(f"Error in on_agent_done: {e}")
        else:
            # Simple logging fallback
            print(f"DONE [{self.agent_id}]: Agent completed with result: {result}")

    def on_agent_start(self, initial_task: str): pass
    def _write_log(self, log_entry: Dict[str, Any]): pass
    def _sanitize_for_json(self, obj: Any): pass 

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        print(f"DEBUG [{self.agent_id}]: {message}")
        if hasattr(self.logger, 'debug'):
            self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        print(f"INFO [{self.agent_id}]: {message}")
        if hasattr(self.logger, 'info'):
            self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        print(f"WARNING [{self.agent_id}]: {message}")
        if hasattr(self.logger, 'warning'):
            self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        print(f"ERROR [{self.agent_id}]: {message}")
        if hasattr(self.logger, 'error'):
            self.logger.error(message, **kwargs)
