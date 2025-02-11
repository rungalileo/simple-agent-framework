from typing import List, Dict, Any, Tuple
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from agent_framework.agent import Agent
from agent_framework.state import AgentState
from agent_framework.llm.models import LLMMessage
from .tools.event_finder import EventFinderTool
from .tools.weather_retriever import WeatherRetrieverTool
from .tools.restaurant_recommender import RestaurantRecommenderTool
from .tools.itinerary_builder import ItineraryBuilderTool
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
        # Itinerary builder
        self.tool_registry.register(
            metadata=ItineraryBuilderTool.get_metadata(),
            implementation=ItineraryBuilderTool
        )

        self._setup_logger(logger=self.logger)

    async def _format_result(self, task: str, results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Format the final result showing the connection between events, weather, and dining"""
        # Since itinerary_builder is the last tool executed and contains the complete narrative,
        # we should use its output as the final result
        for tool_name, result in results:
            if tool_name == "itinerary_builder":
                output = []
                
                # Add weather considerations if available
                weather_considerations = result.get("weather_considerations", {})
                if weather_considerations:
                    output.append("🌤 Weather Considerations:")
                    output.append(f"• Overall: {weather_considerations.get('overall_assessment', 'Not available')}")
                    if adaptations := weather_considerations.get("adaptations", []):
                        output.append("• Adaptations:")
                        for adaptation in adaptations:
                            output.append(f"  - {adaptation}")
                    output.append("")  # Add spacing
                
                # Add the main itinerary narrative
                output.append("📋 Detailed Itinerary:")
                output.append(result.get("itinerary", "Error: No itinerary was generated"))
                output.append("")  # Add spacing
                
                # Add events section with weather justifications
                events = result.get("events", [])
                if events:
                    output.append("🎯 Selected Events and Weather Considerations:")
                    for event in events:
                        output.append(f"\n• {event.get('name', 'Unnamed Event')}")
                        if justification := event.get('weather_justification'):
                            output.append(f"  ↳ {justification}")
                
                # Add restaurants section with pairing reasons
                restaurants = result.get("restaurants", [])
                if restaurants:
                    output.append("\n🍽 Restaurant Pairings and Reasoning:")
                    for restaurant in restaurants:
                        output.append(f"\n• {restaurant.get('name', 'Unnamed Restaurant')}")
                        if reason := restaurant.get('pairing_reason'):
                            output.append(f"  ↳ {reason}")
                
                return "\n".join(output)
        
        # If we didn't find the itinerary builder result, return an error
        return "Error: No itinerary was generated. Please try again."
