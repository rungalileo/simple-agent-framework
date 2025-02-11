from typing import List, Dict, Any, Tuple
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from agent_framework.agent import Agent
from agent_framework.state import AgentState
from agent_framework.llm.models import LLMMessage
from .tools.event_finder import EventFinderTool
from .tools.weather_retriever import WeatherRetrieverTool
from .tools.restaurant_recommender import RestaurantRecommenderTool
from .logging.GalileoAgentLogger import GalileoAgentLogger

class TravelAgent(Agent):
    """Agent that helps users find weather-appropriate events and matching restaurants"""
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
        
        self.logger = GalileoAgentLogger(agent_id=self.agent_id)
        self._register_tools()

    def _create_planning_prompt(self, task: str) -> List[LLMMessage]:
        """Create a planning prompt for travel planning"""
        tools_description = "\n".join([
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Tags: {', '.join(tool.tags)}\n"
            f"Input Schema: {tool.input_schema}\n"
            f"Output Schema: {tool.output_schema}\n"
            for tool in self.tool_registry.get_all_tools().values()
        ])

        # Get the planning template
        template = self.template_env.get_template("planning.j2")
        
        # Create context with tools description and task
        context = {
            "tools_description": tools_description,
            "task": task
        }
        
        # Render the template
        system_content = template.render(**context)
        
        return [
            LLMMessage(role="system", content=system_content),
            LLMMessage(
                role="user", 
                content=f"Plan a travel itinerary for {task} with events, weather-appropriate activities, and restaurant recommendations."
            )
        ]

    def _register_tools(self) -> None:
        """Register all tools with the registry"""
        # Event finder
        self.tool_registry.register(
            metadata=EventFinderTool.get_metadata(),
            implementation=EventFinderTool
        )   
        # Weather retriever
        self.tool_registry.register(
            metadata=WeatherRetrieverTool.get_metadata(),
            implementation=WeatherRetrieverTool
        )
        # Restaurant recommender
        self.tool_registry.register(
            metadata=RestaurantRecommenderTool.get_metadata(),
            implementation=RestaurantRecommenderTool
        )

        self._setup_logger(logger=self.logger)

    async def _format_result(self, task: str, results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Format the final result showing the connection between events, weather, and dining"""
        output = []
        events_data = None
        weather_data = None
        restaurant_data = None

        # Collect data from each tool
        for tool_name, result in results:
            if tool_name == "event_finder":
                events_data = result
            elif tool_name == "weather_retriever":
                weather_data = result
            elif tool_name == "restaurant_recommender":
                restaurant_data = result

        if not all([events_data, weather_data, restaurant_data]):
            return "Incomplete data to provide recommendations"

        # Format selected events with weather context
        output.append("🎯 Selected Events (Weather-Appropriate):")
        for event in events_data.get("events", []):
            output.append(f"\n• {event.get('name')}")
            output.append(f"  📅 {event.get('date')} at {event.get('time')}")
            output.append(f"  📍 {event.get('venue', {}).get('name')}")
            output.append(f"  🌤 Weather: {weather_data.get('weather_condition')}, {weather_data.get('temperature')}°C")
            output.append(f"  🌧 Precipitation Chance: {weather_data.get('precipitation_chance')}%")

        # Format restaurant recommendations
        output.append("\n\n🍽 Recommended Restaurants (Matching Your Activities):")
        for restaurant in restaurant_data.get("restaurants", [])[:3]:  # Top 3 recommendations
            output.append(f"\n• {restaurant.get('name')}")
            output.append(f"  ⭐ {restaurant.get('rating')} stars ({restaurant.get('review_count')} reviews)")
            output.append(f"  💰 {restaurant.get('price_level', 'Price N/A')}")
            output.append(f"  🍳 {', '.join(restaurant.get('cuisine_types', []))}")
            output.append(f"  📍 {restaurant.get('location', {}).get('address')}")

        return "\n".join(output)
