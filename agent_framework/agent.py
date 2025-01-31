from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4
from datetime import datetime
import json
from .utils.logging import AgentLogger
from .utils.tool_registry import ToolRegistry
from rich.panel import Panel
from rich.json import JSON
from rich.console import Console

from .models import (
    AgentMetadata, TaskExecution, ExecutionStep, ToolCall,
    Tool, ToolSelectionCriteria, ToolSelectionReasoning,
    VerbosityLevel, TaskAnalysis, ToolContext, ToolHooks, ToolSelectionHooks, AgentConfig
)
from .llm.base import LLMProvider
from .llm.models import LLMMessage, LLMConfig, ToolSelectionOutput
from .llm.openai_provider import OpenAIProvider
from .utils.formatting import (
    display_task_header, display_analysis, display_chain_of_thought,
    display_execution_plan, display_tool_result, display_final_result,
    display_error
)
from .exceptions import ToolNotFoundError, ToolExecutionError

class Agent(ABC):
    """Base class for all agents in the framework"""
    
    def __init__(
        self,
        *args,
        verbosity: VerbosityLevel = VerbosityLevel.LOW,
        logger: Optional[AgentLogger] = None,
        tool_selection_hooks: Optional[ToolSelectionHooks] = None,
        metadata: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[LLMProvider] = None,
        **kwargs
    ):
        self.agent_id = str(uuid4())
        self.config = AgentConfig(
            verbosity=verbosity,
            logger=logger,
            tool_selection_hooks=tool_selection_hooks,
            metadata=metadata or {}
        )
        self.llm_provider = llm_provider
        self.tool_registry = ToolRegistry()
        self.current_task: Optional[TaskExecution] = None
        self.state: Dict[str, Any] = {}
        self.message_history: List[Dict[str, Any]] = []

    def log(self, message: str, level: VerbosityLevel = VerbosityLevel.LOW) -> None:
        """Log a message if verbosity level is sufficient"""
        if self.config.verbosity.value >= level.value:
            print(message)

    def _create_tool_context(self, tool_name: str, inputs: Dict[str, Any]) -> ToolContext:
        """Create a context object for tool execution"""
        if not self.current_task:
            raise ValueError("No active task")
            
        return ToolContext(
            task=self.current_task.input,
            tool_name=tool_name,
            inputs=inputs,
            previous_tools=[step.tool_name for step in self.current_task.steps],
            previous_results=[step.result for step in self.current_task.steps if step.result],
            previous_errors=[step.error for step in self.current_task.steps if step.error],
            message_history=self.message_history.copy(),
            agent_id=self.agent_id,
            task_id=self.current_task.task_id,
            start_time=self.current_task.start_time,
            metadata=self.config.metadata
        )

    async def call_tool(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        execution_reasoning: str,
        context: Dict[str, Any],
        selection_criteria: Optional[ToolSelectionCriteria] = None
    ) -> Dict[str, Any]:
        """Execute a tool and log the call with selection reasoning"""
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool_context = self._create_tool_context(tool_name, inputs)
        
        try:
            # Call before_execution hook if available
            if tool.hooks:
                await tool.hooks.before_execution(tool_context)
            
            # Execute the tool using registry
            result = await self._execute_tool(tool_name, inputs)
            
            # Record the execution
            self.message_history.append({
                "role": "tool",
                "tool_name": tool_name,
                "inputs": inputs,
                "result": result,
                "reasoning": execution_reasoning,
                "timestamp": datetime.now()
            })
            
            # Call after_execution hook if available
            if tool.hooks:
                await tool.hooks.after_execution(tool_context, result)
            
            # Display if high verbosity
            # if self.config.verbosity == VerbosityLevel.HIGH:
            #     console = Console()
                
            #     # Convert Pydantic models to dicts for JSON serialization
            #     json_context = {k: v.model_dump() if hasattr(v, 'model_dump') else v 
            #                   for k, v in context.items()}
                
            #     content = [
            #         f"[bold]Tool:[/bold] {tool_name}",
            #         "[bold]Inputs:[/bold]",
            #         console.print(json.dumps(inputs, indent=2), style="cyan"),
            #         "[bold]Execution Reasoning:[/bold]",
            #         execution_reasoning,
            #         "[bold]Context:[/bold]",
            #         console.print(json.dumps(json_context, indent=2), style="cyan")
            #     ]
                
            #     console.print(Panel(
            #         "\n".join(str(line) for line in content),
            #         title="Tool Execution",
            #         border_style="blue"
            #     ))
            
            return result
            
        except Exception as e:
            # Call after_execution hook with error if available
            if tool.hooks:
                await tool.hooks.after_execution(tool_context, None, error=e)
            raise

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
        
        # Log the prompt and context
        if self.config.logger:
            self.config.logger.info(
                "Creating tool selection prompt",
                prompt_messages=messages,
                context=context,
                criteria=criteria.model_dump(),
                available_tools=[tool.name for tool in available_tools]
            )
        
        try:
            # Try using structured output with function calling
            if isinstance(self.llm_provider, OpenAIProvider):
                selection_output = await self.llm_provider.generate_structured(
                    messages,
                    ToolSelectionOutput,
                    self.llm_provider.config
                )
                
                # Log the LLM response
                if self.config.logger:
                    self.config.logger.info(
                        "Received tool selection response",
                        selection_output=selection_output.model_dump(),
                        task_id=self.current_task.task_id if self.current_task else None
                    )
                
                return (
                    selection_output.selected_tools,
                    selection_output.confidence,
                    selection_output.reasoning_steps
                )
        except Exception as e:
            if self.config.logger:
                self.config.logger.error(
                    "Failed to generate structured tool selection",
                    error=str(e),
                    task_id=self.current_task.task_id if self.current_task else None
                )
            raise

    async def select_tool(
        self,
        context: Dict[str, Any],
        criteria: Optional[ToolSelectionCriteria] = None
    ) -> ToolSelectionReasoning:
        """Select the most appropriate tools based on context and criteria"""
        criteria = criteria or self.tool_selection_criteria
        reasoning = ToolSelectionReasoning(
            context=context,
            considered_tools=list(self.tool_registry.get_all_tools().keys()),
            selection_criteria=criteria,
            reasoning_steps=[],
            selected_tools=[],
            confidence_score=0.0
        )
        
        if self.llm_provider:
            selected_tools, confidence, steps = await self._select_tool_with_llm(
                context,
                criteria,
                list(self.tool_registry.get_all_tools().values())
            )
            
            if self.config.verbosity == VerbosityLevel.HIGH:
                self.log("\nTool Selection Process:", VerbosityLevel.HIGH)
                self.log(f"Context: {context}", VerbosityLevel.HIGH)
                self.log(f"Available Tools: {list(self.tool_registry.get_all_tools().keys())}", VerbosityLevel.HIGH)
                self.log("\nReasoning Steps:", VerbosityLevel.HIGH)
                for step in steps:
                    self.log(f"- {step}", VerbosityLevel.HIGH)
                self.log(f"\nSelected Tools: {selected_tools} (Confidence: {confidence:.2f})", VerbosityLevel.HIGH)
        else:
            selected_tools, confidence, steps = self._select_tool(
                context,
                criteria,
                list(self.tool_registry.get_all_tools().values())
            )
            
            if self.config.verbosity == VerbosityLevel.HIGH:
                self.log("\nFallback Tool Selection:", VerbosityLevel.HIGH)
                self.log(f"Selected Tools: {selected_tools} (Confidence: {confidence:.2f})", VerbosityLevel.HIGH)
        
        reasoning.selected_tools = selected_tools
        reasoning.confidence_score = confidence
        reasoning.reasoning_steps = steps
        
        # Call after_selection hook if available
        if self.config.tool_selection_hooks:
            await self.config.tool_selection_hooks.after_selection(
                self._create_tool_context("", {}),
                selected_tools,
                confidence,
                steps
            )
        
        return reasoning

    def _select_tool(
        self,
        context: Dict[str, Any],
        criteria: ToolSelectionCriteria,
        available_tools: List[Tool]
    ) -> tuple[List[str], float, List[str]]:
        """
        Default fallback tool selection logic.
        Returns: (selected_tool_names, confidence_score, reasoning_steps)
        """
        # Default to selecting first available tool with low confidence
        if available_tools:
            return (
                [available_tools[0].name],
                0.5,
                ["Fallback selection: using first available tool"]
            )
        return ([], 0.0, ["No tools available"])

    async def _execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given inputs"""
        tool_impl = self.tool_registry.get_implementation(tool_name)
        if not tool_impl:
            raise ToolNotFoundError(f"No implementation found for tool: {tool_name}")
            
        try:
            tool_instance = tool_impl()
            result = await tool_instance.execute(**inputs)
            
            # Store result in state
            self.state.set_tool_result(tool_name, result)
            
            return result
        except Exception as e:
            raise ToolExecutionError(tool_name, e)

    def _create_planning_prompt(self, task: str) -> List[LLMMessage]:
        """Create prompt for task planning"""
        tools_description = "\n".join([
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Tags: {', '.join(tool.tags)}\n"
            f"Input Schema: {tool.input_schema}\n"
            f"Output Schema: {tool.output_schema}\n"
            for tool in self.tool_registry.get_all_tools().values()
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
        
        # Log the planning prompt
        if self.config.logger:
            self.config.logger.info(
                "Creating task planning prompt",
                prompt_messages=messages,
                task=task,
                task_id=self.current_task.task_id if self.current_task else None
            )
        
        if self.config.verbosity == VerbosityLevel.HIGH:
            display_task_header(task)

        try:
            plan = await self.llm_provider.generate_structured(
                messages,
                TaskAnalysis,
                self.llm_provider.config
            )
            
            # Log the planning response
            if self.config.logger:
                self.config.logger.info(
                    "Received task planning response",
                    plan=plan.model_dump(),
                    task_id=self.current_task.task_id if self.current_task else None
                )
            
            if self.config.verbosity == VerbosityLevel.HIGH:
                display_analysis(plan.input_analysis)
                display_chain_of_thought(plan.chain_of_thought)
                display_execution_plan(plan.execution_plan)
            
            return plan
        except Exception as e:
            if self.config.logger:
                self.config.logger.error(
                    "Failed to generate task plan",
                    error=str(e),
                    task=task,
                    task_id=self.current_task.task_id if self.current_task else None
                )
            if self.config.verbosity == VerbosityLevel.HIGH:
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
            # Create a plan using chain of thought reasoning
            plan = await self.plan_task(task)
            
            # Execute each step in the plan
            results = []
            for step in plan.execution_plan:
                tool_name = step["tool"]
                if not self.tool_registry.get_tool(tool_name):
                    raise ToolNotFoundError(f"Tool {tool_name} not found")
                
                # Map inputs for the tool
                inputs = await self._map_inputs_to_tool(tool_name, task, step.get("input_mapping", {}))
                
                # Execute the tool
                result = await self.call_tool(
                    tool_name=tool_name,
                    inputs=inputs,
                    execution_reasoning=step["reasoning"],
                    context={"task": task, "plan": plan}
                )
                
                if self.config.verbosity == VerbosityLevel.HIGH:
                    display_tool_result(tool_name, result)
                
                results.append((tool_name, result))
            
            # Format final result
            result = await self._format_result(task, results)
            self.current_task.output = result
            
            if self.config.verbosity == VerbosityLevel.HIGH:
                display_final_result(result)
            
            return result
            
        except Exception as e:
            self.current_task.error = str(e)
            self.current_task.status = "failed"
            
            if self.config.verbosity == VerbosityLevel.HIGH:
                display_error(str(e))
            
            raise
        finally:
            self.current_task.end_time = datetime.now()
            if self.current_task.status == "in_progress":
                self.current_task.status = "completed"

    @abstractmethod
    async def _format_result(self, task: str, results: List[tuple[str, Dict[str, Any]]]) -> str:
        """Format the final result from tool executions"""
        pass

    async def _map_inputs_to_tool(self, tool_name: str, task: str, input_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map inputs based on tool schema"""
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool {tool_name} not found")
        
        # Get the tool's input schema
        required_inputs = tool.input_schema.get("properties", {})
        
        # If there's an explicit mapping from the LLM, use it
        if input_mapping:
            return {k: self.state.get_variable(v, v) for k, v in input_mapping.items()}
        
        # Try to map inputs based on schema and state
        mapped_inputs = {}
        for input_name, input_schema in required_inputs.items():
            # Check for referenced tool outputs in schema
            if "$ref" in input_schema:
                ref_tool = input_schema.get("$ref").split("/")[-1]  # Get referenced type name
                # Look for any tool result that matches this type
                for tool_name, result in self.state.tool_results.items():
                    if result and isinstance(result, dict):  # Basic type check
                        mapped_inputs[input_name] = result
                        break
            # Otherwise try direct mapping
            elif self.state.has_tool_result(input_name):
                mapped_inputs[input_name] = self.state.get_tool_result(input_name)
            elif self.state.has_variable(input_name):
                mapped_inputs[input_name] = self.state.get_variable(input_name)
            elif input_schema.get("type") == "string":
                mapped_inputs[input_name] = task
        
        if mapped_inputs:
            return mapped_inputs
            
        raise ValueError(f"Could not map inputs for tool {tool_name}")