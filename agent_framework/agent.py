from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4
from datetime import datetime
import json

from .models import (
    AgentMetadata, TaskExecution, ExecutionStep, ToolCall,
    Tool, ToolSelectionCriteria, ToolSelectionReasoning,
    VerbosityLevel, TaskAnalysis
)
from .llm.base import LLMProvider
from .llm.models import LLMMessage, LLMConfig, ToolSelectionOutput
from .llm.openai_provider import OpenAIProvider
from .utils.formatting import (
    display_task_header, display_analysis, display_chain_of_thought,
    display_execution_plan, display_tool_result, display_final_result,
    display_error
)

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

    async def _execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of tool execution logic.
        
        This is the default implementation that executes tools based on their registered implementations.
        Override this method if you need custom tool execution logic.
        """
        if tool_name not in self.tool_implementations:
            raise ValueError(f"Tool {tool_name} not registered")
        
        implementation = self.tool_implementations[tool_name]
        return await implementation(**inputs)

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
            "You are an intelligent task planning system. Your role is to analyze tasks and create detailed execution plans.\n\n"
            "You MUST provide a complete response with ALL of the following components:\n\n"
            "1. input_analysis: A thorough analysis of the task requirements and constraints\n"
            "2. available_tools: List of all tools that could potentially be used\n"
            "3. tool_capabilities: A mapping of each available tool to its key capabilities\n"
            "4. execution_plan: A list of steps, where each step has:\n"
            "   - tool: The name of the tool to use\n"
            "   - reasoning: Why this tool was chosen for this step\n"
            "5. requirements_coverage: How each requirement is covered by which tools\n"
            "6. chain_of_thought: Your step-by-step reasoning process\n\n"
            f"Available Tools:\n{tools_description}\n\n"
            "Your response MUST be a JSON object with this EXACT structure:\n"
            "{\n"
            '  "input_analysis": "detailed analysis of the task",\n'
            '  "available_tools": ["tool1", "tool2"],\n'
            '  "tool_capabilities": {\n'
            '    "tool1": ["capability1", "capability2"],\n'
            '    "tool2": ["capability3"]\n'
            "  },\n"
            '  "execution_plan": [\n'
            '    {"tool": "tool1", "reasoning": "why tool1 is used"},\n'
            '    {"tool": "tool2", "reasoning": "why tool2 is used"}\n'
            "  ],\n"
            '  "requirements_coverage": {\n'
            '    "requirement1": ["tool1"],\n'
            '    "requirement2": ["tool1", "tool2"]\n'
            "  },\n"
            '  "chain_of_thought": [\n'
            '    "step 1 reasoning",\n'
            '    "step 2 reasoning"\n'
            "  ]\n"
            "}\n\n"
            "Ensure ALL fields are present and properly formatted. Missing fields will cause errors."
        )

        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Task: {task}\n\nAnalyze this task and create a complete execution plan with ALL required fields."
            )
        ]

    async def plan_task(self, task: str) -> TaskAnalysis:
        """Create an execution plan for the task using chain of thought reasoning"""
        if not self.llm_provider:
            raise RuntimeError("LLM provider not configured")

        messages = self._create_planning_prompt(task)
        
        if self.verbosity == VerbosityLevel.HIGH:
            display_task_header(task)

        try:
            plan = await self.llm_provider.generate_structured(
                messages,
                TaskAnalysis,
                self.llm_provider.config
            )
            
            if self.verbosity == VerbosityLevel.HIGH:
                display_analysis(plan.input_analysis)
                display_chain_of_thought(plan.chain_of_thought)
                display_execution_plan(plan.execution_plan)
            
            return plan
        except Exception as e:
            if self.verbosity == VerbosityLevel.HIGH:
                display_error(str(e))
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
                
                # Get the tool's input schema
                tool = self.tools[tool_name]
                required_inputs = tool.input_schema.get("properties", {})
                
                # If the tool expects a single text input, use the task
                if len(required_inputs) == 1 and list(required_inputs.keys())[0] in ["text", "input", "content"]:
                    inputs = {list(required_inputs.keys())[0]: task}
                # If the tool expects a location, use the task as location
                elif len(required_inputs) == 1 and "location" in required_inputs:
                    inputs = {"location": task}
                # For other cases, let the agent implementation handle input mapping
                else:
                    inputs = await self._map_inputs_to_tool(tool_name, task, step.get("input_mapping", {}))
                
                result = await self.call_tool(
                    tool_name=tool_name,
                    inputs=inputs,
                    execution_reasoning=step["reasoning"],
                    context={"task": task, "plan": plan}
                )
                
                if self.verbosity == VerbosityLevel.HIGH:
                    display_tool_result(tool_name, result)
                
                results.append((tool_name, result))
            
            # Create markdown output
            markdown_output = []
            markdown_output.append("# Task Analysis and Results\n")
            markdown_output.append(f"## Input Analysis\n{plan.input_analysis}\n")
            markdown_output.append("## Tool Results\n")
            
            for tool_name, result in results:
                markdown_output.append(f"### {tool_name}\n")
                if isinstance(result, (dict, list)):
                    markdown_output.append("```json\n")
                    markdown_output.append(json.dumps(result, indent=2))
                    markdown_output.append("\n```\n")
                else:
                    markdown_output.append(str(result) + "\n")
            
            combined_result = "\n".join(markdown_output)
            self.current_task.output = combined_result
            
            if self.verbosity == VerbosityLevel.HIGH:
                display_final_result(combined_result)
            
            return combined_result
                
        except Exception as e:
            self.current_task.error = str(e)
            self.current_task.status = "failed"
            
            if self.verbosity == VerbosityLevel.HIGH:
                display_error(str(e))
            
            raise
        finally:
            self.current_task.end_time = datetime.now()
            if self.current_task.status == "in_progress":
                self.current_task.status = "completed"

    async def _map_inputs_to_tool(self, tool_name: str, task: str, input_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map task input to tool-specific inputs. Override this in agent implementations."""
        # Default implementation for backward compatibility
        return {"text": task}

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