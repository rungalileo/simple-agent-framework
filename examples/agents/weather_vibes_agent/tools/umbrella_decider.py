from typing import Dict, Any
from agent_framework.tools.base import BaseTool
from agent_framework.models import ToolError
from .schemas import (
    UmbrellaDeciderInput, 
    UmbrellaDeciderOutput,
    UmbrellaDeciderMetadata,
    WeatherRetrieverOutput
)

class UmbrellaDeciderTool(BaseTool):
    """Tool for deciding if an umbrella is needed"""
    
    name = "umbrella_decider"
    description = "Decide if an umbrella is needed based on weather data"
    tags = ["decision", "weather"]
    input_schema = UmbrellaDeciderInput.model_json_schema()
    output_schema = UmbrellaDeciderOutput.model_json_schema()
    metadata = UmbrellaDeciderMetadata
    
    async def execute(self, weather_data: Dict[str, Any]) -> bool | ToolError:
        """Execute the tool with given inputs"""
        # Convert dict to Pydantic model
        weather_data = WeatherRetrieverOutput(**weather_data)
        
        # Decision logic using validated model
        needs_umbrella = (
            weather_data.precipitation_chance > 30 or
            "rain" in weather_data.weather_condition.lower()
        )
        
        return UmbrellaDeciderOutput(needs_umbrella=needs_umbrella).needs_umbrella 