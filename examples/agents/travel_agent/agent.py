from typing import Optional, List, Dict, Any
from agent_framework.agent import Agent
from agent_framework.models import AgentMetadata
from .tools.event_finder import EventFinderTool
from .tools.weather_retriever import WeatherRetrieverTool
from .tools.restaurant_recommender import RestaurantFinderTool

class TravelAgent(Agent):
    """Agent that helps users find events, weather, and restaurants in a given location"""

    def __init__(
        self,
        verbosity: int = 0,
        logger: Optional[Any] = None,
        tool_selection_hooks: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[str] = None
    ):
        # Initialize base agent
        super().__init__(
            verbosity=verbosity,
            logger=logger,
            tool_selection_hooks=tool_selection_hooks,
            metadata=metadata,
            llm_provider=llm_provider
        )

        # Register tools
        self.register_tool(EventFinderTool())
        self.register_tool(WeatherRetrieverTool())
        self.register_tool(RestaurantFinderTool())

    @classmethod
    def get_metadata(cls) -> AgentMetadata:
        """Get agent metadata"""
        return AgentMetadata(
            name="travel_agent",
            description=(
                "An agent that helps users find events, weather, and restaurants in a given location. "
                "It can search for various events, weather, and restaurants. The agent provides detailed information including "
                "durations, distances, transfers, and step-by-step instructions."
            ),
            version="1.0.0",
            author="Galileo AI",
            website="https://galileo.ai",
            capabilities=[
                "Find events, weather, and restaurants in a given location",
                "Provide detailed information including durations, distances, transfers, and step-by-step instructions."
            ],
            limitations=[
                "Currently requires coordinates for locations (no address geocoding)",
                "Limited to supported events, weather, and restaurants in the database",
                "May not have real-time updates for all events, weather, and restaurants",
                "Requires API authentication"
            ]
        )

