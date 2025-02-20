from typing import List
from pydantic import BaseModel, Field
from agent_framework.models import ToolMetadata

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
        description="The weather condition, but in bananna terms",
        # examples=["partly cloudy", "light rain", "sunny"]
        examples=["banana", "many bananas", "no bananas"]
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