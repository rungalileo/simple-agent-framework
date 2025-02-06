from typing import Any, Dict, List
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from uuid import uuid4
from datetime import datetime

from agent_framework.agent import Agent
from agent_framework.models import TaskAnalysis, TaskExecution
from agent_framework.llm.models import LLMMessage
from agent_framework.state import AgentState
from agent_framework.exceptions import ToolNotFoundError
from typing import Sequence, Union

from .tools.weather_retriever import WeatherRetrieverTool
from .tools.umbrella_decider import UmbrellaDeciderTool
from .logging.GalileoAgentLogger import GalileoAgentLogger

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
        
        # Set up template environment
        template_dir = Path(__file__).parent / "templates"
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Create logger first
        self.logger = GalileoAgentLogger(agent_id=self.agent_id)
        print(f"Logger created: {self.logger}")
        
        # Then register tools and set up hooks
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
        
        print(f"Setting up logger for {self.logger}")
        self._setup_logger(logger=self.logger)

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