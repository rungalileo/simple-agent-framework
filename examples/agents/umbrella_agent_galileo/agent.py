from typing import Any, Dict, List
from pathlib import Path
from galileo_observe import AgentStep
from jinja2 import Environment, FileSystemLoader

from agent_framework.agent import Agent
from agent_framework.models import TaskAnalysis, VerbosityLevel
from agent_framework.llm.models import LLMMessage
from agent_framework.state import AgentState
from agent_framework.utils.tool_registry import ToolRegistry
from agent_framework.utils.tool_hooks import create_tool_hooks, create_tool_selection_hooks
from agent_framework.exceptions import ToolNotFoundError
from typing import Optional, Sequence, Union

from .tools.weather_retriever import WeatherRetrieverTool
from .tools.umbrella_decider import UmbrellaDeciderTool
from .logging.GalileoAgentLogger import GalileoAgentLogger
from agent_framework.llm.openai_provider import OpenAIProvider
from agent_framework.llm.base import LLMProvider

def format_messages(messages: Sequence[Union[LLMMessage, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Format messages into a list of dictionaries suitable for Galileo
    
    Handles both LLMMessage objects and pre-formatted dictionaries.
    For LLMMessage objects, extracts role and content.
    For dictionaries, passes them through if they're already in the right format.
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
            # If it's a tool message, format it appropriately
            if msg.get('role') == 'tool':
                formatted.append({
                    "role": "tool",
                    "name": msg.get('tool_name', ''),
                    "content": str({
                        "inputs": msg.get('inputs', {}),
                        "result": msg.get('result', {}),
                        "reasoning": msg.get('reasoning', '')
                    })
                })
            else:
                # For other types of messages, keep the essential fields
                formatted.append({
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', str(msg))
                })
    return formatted

class UmbrellaAgent(Agent):
    """Agent that determines if you need an umbrella based on weather forecast"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = AgentState()
        self.logger = None  # Initialize as None first
        
        # Set up template environment
        template_dir = Path(__file__).parent / "templates"
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register tools first
        self._register_tools()
        
        # Create and set logger after tools are registered
        self._setup_logger()

    async def run(self, task: str) -> str:
        """Override run to ensure logger is set up"""
        if self.logger is None:
            self._setup_logger()
        return await super().run(task)
        
    def _setup_logger(self) -> None:
        """Create and set up the logger after tools are registered"""
        self.logger = GalileoAgentLogger(agent_id="umbrella_agent")
        print(self.logger)
        # Set hooks for all registered tools
        for tool in self.tool_registry.list_tools():
            tool.hooks = self.logger.get_tool_hooks()
        
        # Set tool selection hooks
        self.tool_selection_hooks = self.logger.get_tool_selection_hooks()

    def _register_tools(self) -> None:
        """Register all tools with the registry"""
        # Weather retriever
        self.tool_registry.register(
            metadata=WeatherRetrieverTool.get_metadata(),
            implementation=WeatherRetrieverTool
        )
        
        # Umbrella decider
        self.tool_registry.register(
            metadata=UmbrellaDeciderTool.get_metadata(),
            implementation=UmbrellaDeciderTool
        )

    def _create_planning_prompt(self, task: str) -> List[LLMMessage]:
        """Create a custom planning prompt for the umbrella agent"""
        tools_description = "\n".join([
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Tags: {', '.join(tool.tags)}\n"
            f"Input Schema: {tool.input_schema}\n"
            f"Output Schema: {tool.output_schema}\n"
            for tool in self.tool_registry.list_tools()
        ])
        
        # Use agent-specific template
        template = self.template_env.get_template("planning.j2")
        system_prompt = template.render(
            tools_description=tools_description
        )
        
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Location: {task}\n\nAnalyze this location and create a complete weather analysis plan with ALL required fields."
            )
        ]

    async def _format_result(self, task: str, results: List[tuple[str, Dict[str, Any]]]) -> str:
        """Format the final result from tool executions"""
        weather_data = self.state.get_tool_result("weather_retriever")
        umbrella_needed = self.state.get_tool_result("umbrella_decider")
        
        result = "You need an umbrella today!" if umbrella_needed else "No umbrella needed today!"
        result += f"\n\nWeather details for {weather_data['location']}:"
        result += f"\n- Temperature: {weather_data.get('temperature', 'N/A')}°C"
        result += f"\n- Condition: {weather_data.get('weather_condition', 'N/A')}"
        result += f"\n- Chance of rain: {weather_data.get('precipitation_chance', 'N/A')}%"
        print("GOT RESULT")
        print(self.logger)
        if self.logger:
            print(f"Uploading workflow for {self.agent_id}")
            wf = self.logger.get_workflow()
            wf.add_llm(
                input=format_messages(self.message_history) if self.message_history else [],
                output=result,
                model="gpt-4-mini",
                metadata={"agent_id": self.agent_id}
            )
            wf.conclude(output={"result": result})
            await self.logger.upload_workflows()  # Make sure this is awaited
        
        return result

    async def _execute_step(self, step: Dict[str, Any], task: str, plan: TaskAnalysis) -> Any:
        """Execute a single step in the plan"""
        tool_name = step["tool"]
        if not self.tool_registry.get_tool(tool_name):
            raise ToolNotFoundError(f"Tool {tool_name} not found")
        
        # Map inputs for the tool
        inputs = await self._map_inputs_to_tool(tool_name, task, step.get("input_mapping", {}))
        
        # Create tool context once for both calls
        tool_context = self._create_tool_context(tool_name, inputs)
        
        # Log tool selection first
        if self.logger:
            await self.logger.get_tool_selection_hooks().after_selection(
                tool_context,
                tool_name,
                1.0,
                [step["reasoning"]]
            )
        
        # Then execute the tool
        result = await self.call_tool(
            tool_name=tool_name,
            inputs=inputs,
            execution_reasoning=step["reasoning"],
            context={"task": task, "plan": plan}
        )
        
        return result