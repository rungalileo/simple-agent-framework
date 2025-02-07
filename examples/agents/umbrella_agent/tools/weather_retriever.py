import aiohttp
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from agent_framework.tools.base import BaseTool

class WeatherRetrieverTool(BaseTool):
    """Tool for retrieving weather data"""

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get tool metadata"""
        return {
            "name": "weather_retriever",
            "description": "Retrieves current weather data for a given location",
            "tags": ["weather", "location"],
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to get weather for (e.g. 'London, UK')"
                    }
                },
                "required": ["location"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "temperature": {"type": "number"},
                    "weather_condition": {"type": "string"},
                    "precipitation_chance": {"type": "number"}
                }
            }
        }

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
                
                return {
                    "location": data["location"]["name"],
                    "temperature": data["current"]["temp_c"],
                    "weather_condition": data["current"]["condition"]["text"],
                    "precipitation_chance": data["current"].get("precip_mm", 0) * 100  # Convert to percentage
                } 