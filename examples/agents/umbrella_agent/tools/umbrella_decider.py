from typing import Dict, Any
from agent_framework.models import Tool

class UmbrellaDeciderTool:
    """Tool for deciding if an umbrella is needed based on weather data"""
    
    @staticmethod
    def get_tool_definition() -> Tool:
        return Tool(
            name="umbrella_decider",
            description="Decides if an umbrella is needed based on weather data",
            input_schema={
                "type": "object",
                "properties": {
                    "weather_data": {
                        "type": "object",
                        "properties": {
                            "precipitation_chance": {"type": "number"},
                            "weather_condition": {"type": "string"}
                        },
                        "required": ["precipitation_chance", "weather_condition"]
                    }
                },
                "required": ["weather_data"]
            },
            output_schema={
                "type": "boolean",
                "description": "True if umbrella is needed, False otherwise"
            },
            tags=["decision", "weather-analysis"]
        )
    
    @staticmethod
    async def execute(weather_data: Dict[str, Any]) -> bool:
        """
        Decides if an umbrella is needed based on weather conditions
        """
        precipitation_chance = weather_data.get("precipitation_chance", 0)
        weather_condition = weather_data.get("weather_condition", "").lower()
        
        # Return True if:
        # - Precipitation chance is 30% or higher
        # - Weather condition contains rain-related keywords
        rain_keywords = ["rain", "shower", "drizzle", "thunderstorm"]
        
        return (
            precipitation_chance >= 30 or
            any(keyword in weather_condition for keyword in rain_keywords)
        ) 