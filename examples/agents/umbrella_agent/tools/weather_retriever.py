from typing import Dict, Any, ClassVar
from agent_framework.tools.base import BaseTool
import aiohttp
from agent_framework.config import load_config
from .schemas import WeatherRetrieverInput, WeatherRetrieverOutput, WeatherRetrieverMetadata

class WeatherRetrieverTool(BaseTool):
    """Tool for retrieving weather data"""
    
    metadata = WeatherRetrieverMetadata
    
    async def execute(self, location: str) -> Dict[str, Any]:
        """Execute the tool with given inputs"""
        config = load_config()
        api_key = config.get("weather_api_key")
        if not api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables")

        async with aiohttp.ClientSession() as session:
            # WeatherAPI.com forecast endpoint
            url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=1&aqi=no"
            
            async with session.get(url) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise ValueError(f"Weather API error: {error_data.get('error', {}).get('message', 'Unknown error')}")
                
                weather_data = await response.json()
                
                # Extract relevant data from the API response
                current = weather_data.get("current", {})
                forecast = weather_data.get("forecast", {}).get("forecastday", [{}])[0]
                day = forecast.get("day", {})
                
                return {
                    "location": location,
                    "precipitation_chance": day.get("daily_chance_of_rain", 0),
                    "weather_condition": current.get("condition", {}).get("text", "unknown"),
                    "temperature": current.get("temp_c", 0)
                } 