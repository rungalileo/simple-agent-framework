from typing import Any, Dict, List, Tuple
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from agent_framework.agent import Agent
from agent_framework.llm.models import LLMMessage
from agent_framework.state import AgentState
from typing import Sequence, Union

from .tools.weather_retriever import WeatherRetrieverTool
from .tools.umbrella_decider import UmbrellaDeciderTool
from .tools.youtube_weather_vibes import YoutubeWeatherVibesTool
from .logging.GalileoAgentLogger import GalileoAgentLogger

class WeatherVibesAgent(Agent):
    """Agent that determines if you need an umbrella and suggests weather-appropriate YouTube videos"""
    
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

        print(f"Agent ID: {self.agent_id}")
        
        # Initialize logger and register tools
        self.logger = GalileoAgentLogger(agent_id=self.agent_id)
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
        
        # YouTube weather vibes
        self.tool_registry.register(
            metadata=YoutubeWeatherVibesTool.get_metadata(),
            implementation=YoutubeWeatherVibesTool
        )
        
        self._setup_logger(logger=self.logger)

    async def _format_result(self, task: str, results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Format the final result from tool executions"""
        weather_data = self.state.get_tool_result("weather_retriever")
        umbrella_needed = self.state.get_tool_result("umbrella_decider")
        youtube_vibes = self.state.get_tool_result("youtube_weather_vibes")
        
        # Format umbrella decision
        result = "You need an umbrella today!" if umbrella_needed else "No umbrella needed today!"
        result += f"\n\nWeather details for {weather_data['location']}:"
        result += f"\n- Temperature: {weather_data.get('temperature', 'N/A')}°C"
        result += f"\n- Condition: {weather_data.get('weather_condition', 'N/A')}"
        result += f"\n- Chance of rain: {weather_data.get('precipitation_chance', 'N/A')}%"
        
        # Format YouTube video recommendations
        if youtube_vibes and "videos" in youtube_vibes and youtube_vibes["videos"]:
            result += "\n\n🎵 Weather Vibe Videos 🎵"
            result += f"\nSearch query: \"{youtube_vibes.get('search_query', 'N/A')}\""
            
            for i, video in enumerate(youtube_vibes["videos"][:3], 1):  # Show top 3 videos
                result += f"\n\n{i}. {video.get('title', 'Unknown')}"
                result += f"\n   {video.get('video_url', 'No URL available')}"
                result += f"\n   {video.get('description', 'No description')[:100]}..." if video.get('description') else ""
        else:
            result += "\n\nNo weather vibe videos found."
        
        return result 