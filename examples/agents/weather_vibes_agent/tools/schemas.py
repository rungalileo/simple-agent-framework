from typing import List, Dict, Any
from pydantic import BaseModel, Field
from agent_framework.models import ToolMetadata

# Reuse the existing schemas from umbrella_agent
class WeatherRetrieverInput(BaseModel):
    """Input schema for weather retriever tool"""
    location: str = Field(
        description="The location to get weather data for",
        examples=["New York", "London, UK"]
    )

class WeatherRetrieverOutput(BaseModel):
    """Output schema for weather retriever tool"""
    location: str = Field(
        description="The location of the weather data",
        examples=["New York", "London, UK"]
    )
    temperature: float = Field(
        description="The temperature in degrees Celsius",
        examples=[18.5, 20.0, 22.0]
    )
    weather_condition: str = Field(
        description="The weather condition",
        examples=["partly cloudy", "light rain", "sunny"]
    )
    precipitation_chance: float = Field(
        description="The chance of precipitation in percent",
        examples=[20.0, 30.0, 40.0]
    )

class UmbrellaDeciderInput(BaseModel):
    """Input schema for umbrella decider tool"""
    weather_data: WeatherRetrieverOutput = Field(
        description="Weather data from the weather retriever tool"
    )

class UmbrellaDeciderOutput(BaseModel):
    """Output schema for umbrella decider tool"""
    needs_umbrella: bool = Field(
        description="Whether an umbrella is needed based on the weather"
    )

# New schemas for YouTube Weather Vibes tool
class YoutubeWeatherVibesInput(BaseModel):
    """Input schema for YouTube weather vibes tool"""
    weather_condition: str = Field(
        description="The current weather condition",
        examples=["partly cloudy", "light rain", "sunny"]
    )
    temperature: float = Field(
        description="The temperature in degrees Celsius",
        examples=[18.5, 20.0, 22.0]
    )

class VideoInfo(BaseModel):
    """Schema for video information"""
    title: str = Field(
        description="The title of the video"
    )
    description: str = Field(
        description="The description of the video"
    )
    thumbnail_url: str = Field(
        description="URL of the video thumbnail"
    )
    video_url: str = Field(
        description="URL to watch the video"
    )
    video_id: str = Field(
        description="YouTube video ID"
    )

class YoutubeWeatherVibesOutput(BaseModel):
    """Output schema for YouTube weather vibes tool"""
    weather_condition: str = Field(
        description="The weather condition used for the search"
    )
    temperature: float = Field(
        description="The temperature used for the search"
    )
    search_query: str = Field(
        description="The search query used to find videos"
    )
    videos: List[VideoInfo] = Field(
        description="List of videos matching the weather vibe"
    )

class WeatherRetrieverMetadata(ToolMetadata):
    """Metadata for weather retriever tool"""
    name: str = "weather_retriever"
    description: str = "Get weather data for a location"
    tags: List[str] = ["weather", "location"]
    input_schema: dict = WeatherRetrieverInput.model_json_schema()
    output_schema: dict = WeatherRetrieverOutput.model_json_schema()
    examples: List[dict] = [
        {
            "input": {"location": "London, UK"},
            "output": {
                "location": "London, UK",
                "temperature": 18.5,
                "weather_condition": "partly cloudy",
                "precipitation_chance": 20.0
            }
        }
    ]

class UmbrellaDeciderMetadata(ToolMetadata):
    """Metadata for umbrella decider tool"""
    name: str = "umbrella_decider"
    description: str = "Decide if an umbrella is needed based on weather data"
    tags: List[str] = ["decision", "weather"]
    input_schema: dict = UmbrellaDeciderInput.model_json_schema()
    output_schema: dict = UmbrellaDeciderOutput.model_json_schema()
    examples: List[dict] = [
        {
            "input": {
                "weather_data": {
                    "location": "London, UK",
                    "temperature": 18.5,
                    "weather_condition": "light rain",
                    "precipitation_chance": 60.0
                }
            },
            "output": {"needs_umbrella": True}
        }
    ]

class YoutubeWeatherVibesMetadata(ToolMetadata):
    """Metadata for YouTube weather vibes tool"""
    name: str = "youtube_weather_vibes"
    description: str = "Find YouTube videos that match the vibe of the current weather"
    tags: List[str] = ["weather", "youtube", "vibes", "entertainment"]
    input_schema: dict = YoutubeWeatherVibesInput.model_json_schema()
    output_schema: dict = YoutubeWeatherVibesOutput.model_json_schema()
    examples: List[dict] = [
        {
            "input": {
                "weather_condition": "light rain",
                "temperature": 18.5
            },
            "output": {
                "weather_condition": "light rain",
                "temperature": 18.5,
                "search_query": "rainy day relaxing mild pleasant music playlist",
                "videos": [
                    {
                        "title": "Rainy Day Jazz - Relaxing Jazz & Bossa Nova Music",
                        "description": "Relaxing jazz music for rainy days...",
                        "thumbnail_url": "https://i.ytimg.com/vi/example/mqdefault.jpg",
                        "video_url": "https://www.youtube.com/watch?v=example",
                        "video_id": "example"
                    }
                ]
            }
        }
    ] 