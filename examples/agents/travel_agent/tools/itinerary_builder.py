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
            description="Builds a weather-aware, thematically cohesive itinerary based on events and/or restaurants",
            tags=["itinerary", "planning", "organization", "travel", "weather"],
            input_schema={
                "type": "object",
                "properties": {
                    "events": {"type": "array", "items": {"type": "object"}},
                    "restaurants": {"type": "array", "items": {"type": "object"}},
                    "weather_data": {"type": "object"}
                },
                "required": []  # Neither events nor restaurants are required
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

    async def execute(self, events: Optional[List[Dict[str, Any]]] = None, restaurants: Optional[List[Dict[str, Any]]] = None, weather_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build an itinerary based on events and/or restaurants"""
        # Create a prompt for the LLM
        events = events or []  # Use empty list if events is None
        restaurants = restaurants or []  # Use empty list if restaurants is None
        
        prompt = [
            LLMMessage(
                role="system",
                content="""You are an expert travel planner creating engaging and weather-aware itineraries.
                Your task is to create a detailed itinerary that:
                1. Organizes activities chronologically and considers weather conditions
                2. Includes events if provided
                3. Includes restaurants if provided
                4. Pairs events with restaurants when both are available
                5. Provides weather-based recommendations and alternatives
                6. Makes meaningful cultural and thematic connections
                7. Explains the reasoning behind each pairing or selection
                8. For every event mentioned, mention the exact time and date of the event
                
                For example:
                - Event-only: "The outdoor jazz festival is perfect for the sunny afternoon"
                - Restaurant-only: "Given the pleasant evening weather, start at the rooftop soul food restaurant"
                - Combined: "After the indoor art gallery opening, head to the nearby fusion restaurant"
                
                Use an engaging, conversational tone and make all connections clear to the reader.
                Include practical weather-based tips and suggestions throughout the itinerary.
                
                Structure your response as a JSON object with these fields:
                {
                    "itinerary": "The complete narrative...",
                    "events": [{"name": "event name", "weather_justification": "why this works with the weather"}],
                    "restaurants": [{"name": "restaurant name", "pairing_reason": "why this was selected"}],
                    "weather_considerations": {"overall_assessment": "weather impact", "adaptations": ["specific adjustments"]}
                }"""
            ),
            LLMMessage(
                role="user",
                content=f"""Create a weather-aware, thematically cohesive itinerary using:

                {f'Events:\n{json.dumps(events, indent=2)}' if events else 'No events provided'}

                {f'Restaurants:\n{json.dumps(restaurants, indent=2)}' if restaurants else 'No restaurants provided'}

                Weather Conditions:
                {json.dumps(weather_data, indent=2) if weather_data else "Weather data not available"}

                Create an itinerary that considers weather conditions and explains the reasoning behind each choice."""
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
        