from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4
from datetime import datetime

from .models import (
    AgentMetadata, TaskExecution, ExecutionStep, ToolCall,
    Tool, ToolSelectionCriteria, ToolSelectionReasoning,
    VerbosityLevel, TaskAnalysis
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
        verbosity: VerbosityLevel = VerbosityLevel.LOW
    ):
        self.agent_id = agent_id or str(uuid4())
        self.metadata = metadata or AgentMetadata(name=self.__class__.__name__)
        self.state: Dict[str, Any] = {}
        self.current_task: Optional[TaskExecution] = None
        self.tools: Dict[str, Tool] = {}
        self.tool_implementations: Dict[str, Callable[..., Awaitable[Dict[str, Any]]]] = {}
        self.tool_selection_criteria = tool_selection_criteria or ToolSelectionCriteria()
        self.llm_provider = llm_provider
        self.verbosity = verbosity

    def log(self, message: str, level: VerbosityLevel = VerbosityLevel.LOW) -> None:
        """Log a message if verbosity level is sufficient"""
        if self.verbosity.value >= level.value:
            print(message)

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
            "and select the most appropriate tools based on the task requirements and tool capabilities. "
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
                content="Please select the most appropriate tools and provide your reasoning."
            )
        ]

    async def _select_tool_with_llm(
        self,
        context: Dict[str, Any],
        criteria: ToolSelectionCriteria,
        available_tools: List[Tool]
    ) -> tuple[List[str], float, List[str]]:
        """Use LLM to select appropriate tools"""
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
                    self.llm_provider.config
                )
                return (
                    selection_output.selected_tools,
                    selection_output.confidence,
                    selection_output.reasoning_steps
                )
        except Exception:
            # Fall back to regular generation if structured output fails
            pass
            
        # Regular generation with JSON parsing
        response = await self.llm_provider.generate(
            messages,
            self.llm_provider.config
        )
        
        selection_output = ToolSelectionOutput.model_validate_json(
            response.content
        )
        
        return (
            selection_output.selected_tools,
            selection_output.confidence,
            selection_output.reasoning_steps
        )

    async def select_tool(
        self,
        context: Dict[str, Any],
        criteria: Optional[ToolSelectionCriteria] = None
    ) -> ToolSelectionReasoning:
        """Select the most appropriate tools based on context and criteria"""
        criteria = criteria or self.tool_selection_criteria
        reasoning = ToolSelectionReasoning(
            context=context,
            considered_tools=list(self.tools.keys()),
            selection_criteria=criteria,
            reasoning_steps=[],
            selected_tools=[],
            confidence_score=0.0
        )
        
        if self.llm_provider:
            selected_tools, confidence, steps = await self._select_tool_with_llm(
                context,
                criteria,
                list(self.tools.values())
            )
            
            if self.verbosity == VerbosityLevel.HIGH:
                self.log("\nTool Selection Process:", VerbosityLevel.HIGH)
                self.log(f"Context: {context}", VerbosityLevel.HIGH)
                self.log(f"Available Tools: {list(self.tools.keys())}", VerbosityLevel.HIGH)
                self.log("\nReasoning Steps:", VerbosityLevel.HIGH)
                for step in steps:
                    self.log(f"- {step}", VerbosityLevel.HIGH)
                self.log(f"\nSelected Tools: {selected_tools} (Confidence: {confidence:.2f})", VerbosityLevel.HIGH)
        else:
            selected_tools, confidence, steps = self._select_tool(
                context,
                criteria,
                list(self.tools.values())
            )
            
            if self.verbosity == VerbosityLevel.HIGH:
                self.log("\nFallback Tool Selection:", VerbosityLevel.HIGH)
                self.log(f"Selected Tools: {selected_tools} (Confidence: {confidence:.2f})", VerbosityLevel.HIGH)
        
        reasoning.selected_tools = selected_tools
        reasoning.confidence_score = confidence
        reasoning.reasoning_steps = steps
        
        return reasoning

    @abstractmethod
    def _select_tool(
        self,
        context: Dict[str, Any],
        criteria: ToolSelectionCriteria,
        available_tools: List[Tool]
    ) -> tuple[List[str], float, List[str]]:
        """
        Implementation of tool selection logic
        Returns: (selected_tool_names, confidence_score, reasoning_steps)
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
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
            
        if self.verbosity == VerbosityLevel.HIGH:
            self.log(f"\nExecuting Tool: {tool_name}", VerbosityLevel.HIGH)
            self.log(f"Execution Reasoning: {execution_reasoning}", VerbosityLevel.HIGH)
            self.log(f"Inputs: {inputs}", VerbosityLevel.HIGH)
        
        result = await self._execute_tool(tool_name, inputs)
        
        if self.verbosity == VerbosityLevel.HIGH:
            self.log(f"Tool Execution Result: {result}", VerbosityLevel.HIGH)
        
        return result

    def _create_planning_prompt(self, task: str) -> List[LLMMessage]:
        """Create prompt for task planning"""
        tools_description = "\n".join([
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Tags: {', '.join(tool.tags)}\n"
            f"Input Schema: {tool.input_schema}\n"
            f"Output Schema: {tool.output_schema}\n"
            for tool in self.tools.values()
        ])

        system_prompt = (
            "You are an intelligent task planning system. Your role is to:\n"
            "1. Analyze the input task thoroughly\n"
            "2. Identify key requirements and constraints\n"
            "3. Evaluate available tools and their capabilities\n"
            "4. Create a step-by-step execution plan\n"
            "5. Show your chain of thought reasoning\n\n"
            f"Available Tools:\n{tools_description}\n\n"
            "Your response must be a JSON object matching this schema:\n"
            f"{TaskAnalysis.model_json_schema()}\n\n"
            "Think through each step carefully and explain your reasoning."
        )

        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Task: {task}\n\nPlease analyze this task and create an execution plan."
            )
        ]

    async def plan_task(self, task: str) -> TaskAnalysis:
        """Create an execution plan for the task using chain of thought reasoning"""
        if not self.llm_provider:
            raise RuntimeError("LLM provider not configured")

        messages = self._create_planning_prompt(task)
        
        if self.verbosity == VerbosityLevel.HIGH:
            self.log("\nStarting Task Planning:", VerbosityLevel.HIGH)
            self.log(f"Task: {task}", VerbosityLevel.HIGH)
            self.log(f"Available Tools: {list(self.tools.keys())}", VerbosityLevel.HIGH)

        try:
            plan = await self.llm_provider.generate_structured(
                messages,
                TaskAnalysis,
                self.llm_provider.config
            )
            
            if self.verbosity == VerbosityLevel.HIGH:
                self.log("\nTask Analysis:", VerbosityLevel.HIGH)
                self.log(f"Input Analysis: {plan.input_analysis}", VerbosityLevel.HIGH)
                self.log("\nChain of Thought:", VerbosityLevel.HIGH)
                for step in plan.chain_of_thought:
                    self.log(f"- {step}", VerbosityLevel.HIGH)
                self.log("\nExecution Plan:", VerbosityLevel.HIGH)
                for step in plan.execution_plan:
                    self.log(f"- Tool: {step['tool']}", VerbosityLevel.HIGH)
                    self.log(f"  Reasoning: {step['reasoning']}", VerbosityLevel.HIGH)
            
            return plan
        except Exception as e:
            if self.verbosity == VerbosityLevel.HIGH:
                self.log(f"\nError in task planning: {str(e)}", VerbosityLevel.HIGH)
            raise

    async def run(self, task: str) -> str:
        """Execute a task and return the result"""
        self.current_task = TaskExecution(
            task_id=str(uuid4()),
            agent_id=self.agent_id,
            input=task,
            start_time=datetime.now(),
            steps=[]
        )
        
        try:
            # First, create a plan using chain of thought reasoning
            plan = await self.plan_task(task)
            
            # Execute each step in the plan
            results = []
            for step in plan.execution_plan:
                tool_name = step["tool"]
                if tool_name not in self.tools:
                    raise ValueError(f"Tool {tool_name} not found")
                
                result = await self.call_tool(
                    tool_name=tool_name,
                    inputs={"text": task},
                    execution_reasoning=step["reasoning"],
                    context={"task": task, "plan": plan}
                )
                results.append((tool_name, result))
            
            # Combine results from all tools
            if len(results) > 1:
                combined_result = "Task Analysis and Results:\n\n"
                combined_result += f"Input Analysis: {plan.input_analysis}\n\n"
                combined_result += "Tool Results:\n"
                for tool_name, result in results:
                    combined_result += f"\n{tool_name}:\n{result}\n"
                self.current_task.output = combined_result
                return combined_result
            elif results:
                self.current_task.output = results[0][1]
                return results[0][1]
            else:
                self.current_task.output = "No tools were executed"
                return "No tools were executed"
                
        except Exception as e:
            self.current_task.error = str(e)
            self.current_task.status = "failed"
            raise
        finally:
            self.current_task.end_time = datetime.now()
            if self.current_task.status == "in_progress":
                self.current_task.status = "completed"

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