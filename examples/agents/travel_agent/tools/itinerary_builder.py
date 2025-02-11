import aiohttp
import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agent_framework.tools.base import BaseTool
from agent_framework.models import ToolMetadata
from agent_framework.llm.openai_provider import OpenAIProvider
from agent_framework.llm.models import LLMMessage, LLMConfig

class ItineraryOutput(BaseModel):
    """Structured output for the itinerary"""
    itinerary: str = Field(
        default="",
        description="The complete narrative itinerary combining events, weather considerations, and dining recommendations"
    )
    events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of selected events with weather-based justification"
    )
    restaurants: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of restaurants paired with events, including thematic connections"
    )
    weather_considerations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Weather-based planning considerations and adaptations"
    )

class ItineraryBuilderTool(BaseTool):
    """Tool for building an itinerary based on events and restaurants"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize LLM with creative temperature
        llm_config = LLMConfig(
            model="gpt-4",
            temperature=0.7  # More creative for itinerary generation
        )
        self.llm = OpenAIProvider(config=llm_config)

    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """Get tool metadata"""
        return ToolMetadata(
            name="itinerary_builder",
            description="Builds a weather-aware, thematically cohesive itinerary based on events and restaurants",
            tags=["itinerary", "planning", "organization", "travel", "weather"],
            input_schema={
                "type": "object",
                "properties": {
                    "events": {"type": "array", "items": {"type": "object"}},
                    "restaurants": {"type": "array", "items": {"type": "object"}},
                    "weather_data": {"type": "object"}
                },
                "required": ["events", "restaurants"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "itinerary": {"type": "string"},
                    "events": {"type": "array"},
                    "restaurants": {"type": "array"},
                    "weather_considerations": {"type": "object"}
                },
                "required": ["itinerary"]
            }
        )

    async def execute(self, events: List[Dict[str, Any]], restaurants: List[Dict[str, Any]], weather_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build an itinerary based on events, restaurants, and weather conditions"""
        # Create a prompt for the LLM
        prompt = [
            LLMMessage(
                role="system",
                content="""You are an expert travel planner creating engaging and weather-aware itineraries.
                Your task is to create a detailed itinerary that:
                1. Organizes events chronologically and considers weather conditions
                2. Pairs events with thematically appropriate restaurants
                3. Provides weather-based recommendations and alternatives
                4. Makes meaningful cultural and thematic connections
                5. Explains the reasoning behind each pairing
                6. For every event mentioned, mention the excat time and date of the event. 
                
                For example:
                - "Given the sunny weather, the outdoor jazz festival pairs perfectly with the rooftop soul food restaurant"
                - "With light rain expected, we've chosen the indoor Latin music venue and matched it with the cozy Mexican restaurant"
                - "The modern art gallery opening is climate-controlled, making it ideal for the hot afternoon, and pairs naturally with the contemporary fusion restaurant"
                
                Use an engaging, conversational tone and make all connections clear to the reader.
                Include practical weather-based tips and suggestions throughout the itinerary.
                
                Structure your response as a JSON object with these fields:
                {
                    "itinerary": "The complete narrative...",
                    "events": [{"name": "event name", "weather_justification": "why this works with the weather"}],
                    "restaurants": [{"name": "restaurant name", "pairing_reason": "why this pairs well"}],
                    "weather_considerations": {"overall_assessment": "weather impact", "adaptations": ["specific adjustments"]}
                }"""
            ),
            LLMMessage(
                role="user",
                content=f"""Create a weather-aware, thematically cohesive itinerary using:

                Events:
                {json.dumps(events, indent=2)}

                Restaurants:
                {json.dumps(restaurants, indent=2)}

                Weather Conditions:
                {json.dumps(weather_data, indent=2) if weather_data else "Weather data not available"}

                Create an itinerary that considers weather conditions, pairs events with thematically appropriate restaurants,
                and explains the reasoning behind each choice."""
            )
        ]

        try:
            # Generate the itinerary using structured output
            response = await self.llm.generate_structured(
                messages=prompt,
                output_model=ItineraryOutput
            )
            
            return {
                "itinerary": response.itinerary,
                "events": response.events,
                "restaurants": response.restaurants,
                "weather_considerations": response.weather_considerations
            }
            
        except Exception as e:
            print(f"Error generating itinerary: {str(e)}")
            return {
                "itinerary": f"Error: Unable to generate itinerary. {str(e)}",
                "events": [],
                "restaurants": [],
                "weather_considerations": {"error": str(e)}
            }
        