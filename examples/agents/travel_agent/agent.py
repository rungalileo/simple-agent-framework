from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from agent_framework.agent import Agent
from agent_framework.models import AgentMetadata, TaskAnalysis
from agent_framework.state import AgentState
from agent_framework.llm.models import LLMMessage
from .tools.event_finder import EventFinderTool
from .tools.weather_retriever import WeatherRetrieverTool
from .tools.restaurant_recommender import RestaurantRecommenderTool

class TravelAgent(Agent):
    """Agent that helps users find weather-appropriate events and matching restaurants"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = AgentState()
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

        system_prompt = (
            "You are an intelligent travel planning system that helps users find weather-appropriate events "
            "and thematically matching restaurants in their chosen city.\n\n"
            "You MUST follow this exact sequence:\n"
            "1. First, use event_finder to discover events in the target city\n"
            "2. Then, use weather_retriever to check weather conditions\n"
            "3. Finally, use restaurant_recommender to find dining options\n\n"
            "You MUST provide a complete response with ALL of the following components:\n\n"
            "1. input_analysis: Analyze the target city (e.g., 'Houston, TX') and any preferences\n"
            "2. available_tools: List the tools: event_finder, weather_retriever, restaurant_recommender\n"
            "3. tool_capabilities: Map each tool to its capabilities\n"
            "4. execution_plan: MUST be exactly 3 steps in this order:\n"
            "   - First step: Use event_finder with the city name\n"
            "   - Second step: Use weather_retriever with the same city name\n"
            "   - Third step: Use restaurant_recommender with the same city name\n"
            "5. requirements_coverage: How each tool contributes to the experience\n"
            "6. chain_of_thought: Your step-by-step reasoning\n\n"
            f"Available Tools:\n{tools_description}\n\n"
            "Your response MUST be a JSON object with this EXACT structure:\n"
            "{\n"
            '  "input_analysis": "Analysis of the target city",\n'
            '  "available_tools": ["event_finder", "weather_retriever", "restaurant_recommender"],\n'
            '  "tool_capabilities": {\n'
            '    "event_finder": ["discover local events", "filter by date and location"],\n'
            '    "weather_retriever": ["get weather forecasts"],\n'
            '    "restaurant_recommender": ["find matching restaurants"]\n'
            "  },\n"
            '  "execution_plan": [\n'
            '    {"tool": "event_finder", "reasoning": "Find events", "input_mapping": {"location": "Houston, TX"}},\n'
            '    {"tool": "weather_retriever", "reasoning": "Check weather", "input_mapping": {"location": "Houston, TX"}},\n'
            '    {"tool": "restaurant_recommender", "reasoning": "Find restaurants", "input_mapping": {"location": "Houston, TX"}}\n'
            "  ],\n"
            '  "requirements_coverage": {\n'
            '    "events": ["event_finder"],\n'
            '    "weather": ["weather_retriever"],\n'
            '    "dining": ["restaurant_recommender"]\n'
            "  },\n"
            '  "chain_of_thought": [\n'
            '    "First find events in the city",\n'
            '    "Then check weather conditions",\n'
            '    "Finally find matching restaurants"\n'
            "  ]\n"
            "}\n\n"
            "Ensure ALL fields are present and properly formatted. Missing fields will cause errors."
        )

        return [
            LLMMessage(role="system", content=system_prompt),
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
