from typing import Any, Dict, List
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from agent_framework.agent import Agent

from agent_framework.llm.models import LLMMessage
from agent_framework.utils.logging import LoggingToolHooks
from agent_framework.state import AgentState

from .tools.weather_retriever import WeatherRetrieverTool
from .tools.umbrella_decider import UmbrellaDeciderTool

class UmbrellaAgent(Agent):
    """Agent that determines if you need an umbrella based on weather forecast"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = AgentState()
        
        # Set up template environment
        template_dir = Path(__file__).parent / "templates"
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        self._register_tools()
    
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
        
        # Set hooks after registration if logger is available
        if self.config.logger:
            for tool in self.tool_registry.list_tools():
                tool.hooks = LoggingToolHooks(self.config.logger)

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
        
        return result