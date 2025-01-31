from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4
from datetime import datetime

from .models import (
    AgentMetadata, TaskExecution, ExecutionStep, ToolCall,
    Tool, ToolSelectionCriteria, ToolSelectionReasoning
)
from .llm.base import LLMProvider
from .llm.models import LLMMessage, LLMConfig, ToolSelectionOutput
from .llm.openai_provider import OpenAIProvider

class Agent(ABC):
    """Base class for all agents in the framework"""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        metadata: Optional[AgentMetadata] = None,
        tool_selection_criteria: Optional[ToolSelectionCriteria] = None,
        llm_provider: Optional[LLMProvider] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        self.agent_id = agent_id or str(uuid4())
        self.metadata = metadata or AgentMetadata(name=self.__class__.__name__)
        self.state: Dict[str, Any] = {}
        self.current_task: Optional[TaskExecution] = None
        self.tools: Dict[str, Tool] = {}
        self.tool_implementations: Dict[str, Callable[..., Awaitable[Dict[str, Any]]]] = {}
        self.tool_selection_criteria = tool_selection_criteria or ToolSelectionCriteria()
        self.llm_provider = llm_provider
        self.llm_config = llm_config

    def register_tool(
        self,
        tool: Tool,
        implementation: Callable
    ) -> None:
        """Register a tool and its implementation"""
        self.tools[tool.name] = tool
        self.tool_implementations[tool.name] = implementation

    def _create_tool_selection_prompt(
        self,
        context: Dict[str, Any],
        criteria: ToolSelectionCriteria,
        available_tools: List[Tool]
    ) -> List[LLMMessage]:
        """Create prompt for tool selection"""
        task = context.get("task", "")
        requirements = context.get("requirements", [])
        
        tools_description = "\n".join([
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Tags: {', '.join(tool.tags)}\n"
            f"Input Schema: {tool.input_schema}\n"
            f"Output Schema: {tool.output_schema}\n"
            for tool in available_tools
        ])
        
        criteria_description = (
            f"Required tags: {criteria.required_tags}\n"
            f"Preferred tags: {criteria.preferred_tags}\n"
            f"Context requirements: {criteria.context_requirements}\n"
            f"Custom rules: {criteria.custom_rules}"
        )
        
        context_description = f"""
 Task: {task}
 Requirements: {', '.join(requirements)}
 Additional Context:
 {chr(10).join(f'- {k}: {v}' for k, v in context.items() if k not in ['task', 'requirements'])}
 """

        output_schema = ToolSelectionOutput.model_json_schema()
        
        system_prompt = (
            "You are an intelligent tool selection system. Your task is to analyze the input "
            "and select the most appropriate tool based on the task requirements and tool capabilities. "
            "Consider the following aspects in your analysis:\n"
            "1. Task requirements and complexity\n"
            "2. Tool capabilities and limitations\n"
            "3. Input/output compatibility\n"
            "4. Specific context requirements\n\n"
            f"Available Tools:\n{tools_description}\n\n"
            f"Selection Criteria:\n{criteria_description}\n\n"
            f"Context:\n{context_description}\n\n"
            "Provide a thorough analysis of the task and justify your tool selection. "
            "Consider edge cases and potential limitations of each tool.\n\n"
            "Provide your response as a JSON object matching this schema:\n"
            f"{output_schema}\n\n"
            "Ensure your response is valid JSON and matches the schema exactly."
        )
        
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content="Please select the most appropriate tool and provide your reasoning."
            )
        ]

    async def _select_tool_with_llm(
        self,
        context: Dict[str, Any],
        criteria: ToolSelectionCriteria,
        available_tools: List[Tool]
    ) -> tuple[str, float, List[str]]:
        """Use LLM to select appropriate tool"""
        if not self.llm_provider:
            raise RuntimeError("LLM provider not configured")
            
        messages = self._create_tool_selection_prompt(
            context,
            criteria,
            available_tools
        )
        
        try:
            # Try using structured output with function calling
            if isinstance(self.llm_provider, OpenAIProvider):
                selection_output = await self.llm_provider.generate_structured(
                    messages,
                    ToolSelectionOutput,
                    self.llm_config
                )
                return (
                    selection_output.selected_tool,
                    selection_output.confidence,
                    selection_output.reasoning_steps
                )
        except Exception:
            # Fall back to regular generation if structured output fails
            pass
            
        # Regular generation with JSON parsing
        response = await self.llm_provider.generate(
            messages,
            self.llm_config
        )
        
        selection_output = ToolSelectionOutput.model_validate_json(
            response.content
        )
        
        return (
            selection_output.selected_tool,
            selection_output.confidence,
            selection_output.reasoning_steps
        )

    async def select_tool(
        self,
        context: Dict[str, Any],
        criteria: Optional[ToolSelectionCriteria] = None
    ) -> ToolSelectionReasoning:
        """Select the most appropriate tool based on context and criteria"""
        criteria = criteria or self.tool_selection_criteria
        reasoning = ToolSelectionReasoning(
            context=context,
            considered_tools=list(self.tools.keys()),
            selection_criteria=criteria,
            reasoning_steps=[],
            selected_tool="",
            confidence_score=0.0
        )
        
        if self.llm_provider:
            selected_tool, confidence, steps = await self._select_tool_with_llm(
                context,
                criteria,
                list(self.tools.values())
            )
        else:
            selected_tool, confidence, steps = self._select_tool(
                context,
                criteria,
                list(self.tools.values())
            )
        
        reasoning.selected_tool = selected_tool
        reasoning.confidence_score = confidence
        reasoning.reasoning_steps = steps
        
        return reasoning

    @abstractmethod
    def _select_tool(
        self,
        context: Dict[str, Any],
        criteria: ToolSelectionCriteria,
        available_tools: List[Tool]
    ) -> tuple[str, float, List[str]]:
        """
        Implementation of tool selection logic
        Returns: (selected_tool_name, confidence_score, reasoning_steps)
        """
        pass

    async def call_tool(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        execution_reasoning: str,
        context: Optional[Dict[str, Any]] = None,
        selection_criteria: Optional[ToolSelectionCriteria] = None
    ) -> Dict[str, Any]:
        """Execute a tool and log the call with selection reasoning"""
        # Validate tool exists
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered")
            
        # Generate tool selection reasoning if context provided
        selection_reasoning = None
        if context is not None:
            selection_reasoning = await self.select_tool(context, selection_criteria)

        tool_call = ToolCall(
            tool_name=tool_name,
            inputs=inputs,
            execution_reasoning=execution_reasoning,
            selection_reasoning=selection_reasoning
        )
        
        try:
            # Execute the tool
            implementation = self.tool_implementations[tool_name]
            result = await self._execute_tool(tool_name, inputs)
            tool_call.outputs = result
            return result
            
        except Exception as e:
            tool_call.success = False
            tool_call.error = str(e)
            raise
        finally:
            # Log the tool call in the current step
            if self.current_task and self.current_task.steps:
                self.current_task.steps[-1].tool_calls.append(tool_call)

    async def run(self, task: str) -> str:
        """Execute a task and return the result"""
        # Initialize task execution record
        self.current_task = TaskExecution(
            task_id=str(uuid4()),
            agent_id=self.agent_id,
            input=task
        )
        
        try:
            # Execute the task
            result = await self._execute_task(task)
            
            # Update task execution record
            self.current_task.output = result
            self.current_task.status = "completed"
            self.current_task.end_time = datetime.utcnow()
            
            return result
            
        except Exception as e:
            # Log failure and reraise
            self.current_task.status = "failed"
            self.current_task.end_time = datetime.utcnow()
            raise

    @abstractmethod
    async def _execute_task(self, task: str) -> str:
        """Implementation of task execution logic"""
        pass

    def log_step(
        self,
        step_type: str,
        description: str,
        tool_calls: Optional[List[ToolCall]] = None,
        intermediate_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an execution step"""
        if not self.current_task:
            raise RuntimeError("No active task execution")
            
        step = ExecutionStep(
            step_type=step_type,
            description=description,
            tool_calls=tool_calls or [],
            intermediate_state=intermediate_state
        )
        self.current_task.steps.append(step)

    @abstractmethod
    async def _execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of tool execution logic"""
        pass 