import aiohttp
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from agent_framework.tools.base import BaseTool
from agent_framework.models import ToolMetadata
from .schemas import WeatherRetrieverInput, WeatherRetrieverOutput, WeatherRetrieverMetadata
class WeatherRetrieverTool(BaseTool):
    """Tool for retrieving weather data"""

    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """Get tool metadata"""
        return ToolMetadata(
            name="weather_retriever",
            description="Retrieves current weather data for a given location",
            tags=["weather", "location"],
            input_schema=WeatherRetrieverInput.model_json_schema(),
            output_schema=WeatherRetrieverOutput.model_json_schema(),
        )

    async def execute(self, location: str) -> Dict[str, Any]:
        """Get weather data for location"""
        load_dotenv()
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            raise ValueError("WEATHER_API_KEY environment variable is required")

        # API endpoint
        url = "http://api.weatherapi.com/v1/current.json"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={
                    "key": api_key,
                    "q": location,
                    "aqi": "no"
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"Weather API error: {await response.text()}")
                    
                data = await response.json()

                # Simulate an error
                # return {
                #     "location": "Simulated error",
                #     "temperature": 0.0,
                #     "weather_condition": "Simulated error",
                #     "precipitation_chance": 0.0                    
                # }
                
                return {
                    "location": data["location"]["name"],
                    "temperature": data["current"]["temp_c"],
                    "weather_condition": data["current"]["condition"]["text"],
                    "precipitation_chance": data["current"].get("precip_mm", 0) * 100  # Convert to percentage
                } 