from typing import Any, Dict, List
from agent_framework.agent import Agent
from agent_framework.models import Tool, ToolSelectionCriteria
from agent_framework.llm.models import LLMConfig
from agent_framework.llm.openai_provider import OpenAIProvider
from agent_framework.config import load_config
from examples.agents.umbrella_agent.tools.weather_retriever import WeatherRetrieverTool
from examples.agents.umbrella_agent.tools.umbrella_decider import UmbrellaDeciderTool

class UmbrellaAgent(Agent):
    """Agent that determines if you need an umbrella based on weather forecast"""
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        # Load configuration
        config = load_config()
        
        # Configure LLM provider
        llm_config = LLMConfig(
            model="gpt-4",
            temperature=0.1  # Lower temperature for more consistent decision making
        )
        llm_provider = OpenAIProvider(
            config=llm_config,
            api_key=config["openai_api_key"]
        )
        
        super().__init__(
            *args,
            llm_provider=llm_provider,
            llm_config=llm_config,
            **kwargs
        )
        
        # Register available tools
        self.register_tool(
            WeatherRetrieverTool.get_tool_definition(),
            lambda location: WeatherRetrieverTool.execute(location)
        )
        
        self.register_tool(
            UmbrellaDeciderTool.get_tool_definition(),
            lambda weather_data: UmbrellaDeciderTool.execute(weather_data)
        )

    async def _execute_task(self, location: str) -> str:
        # Log the initial step
        self.log_step(
            step_type="task_received",
            description=f"Checking weather for location: {location}",
            intermediate_state={"location": location}
        )
        
        # Get weather data
        weather_result = await self.call_tool(
            tool_name="weather_retriever",
            inputs={"location": location},
            execution_reasoning="Need to get weather data for the location"
        )
        
        # Decide if umbrella is needed
        umbrella_decision = await self.call_tool(
            tool_name="umbrella_decider",
            inputs={"weather_data": weather_result},
            execution_reasoning="Need to determine if umbrella is needed based on weather"
        )
        
        # Log final step and return decision
        self.log_step(
            step_type="completion",
            description="Umbrella decision made",
            intermediate_state={"needs_umbrella": umbrella_decision}
        )
        
        return "You need an umbrella today!" if umbrella_decision else "No umbrella needed today!"

    async def _execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of tool execution logic"""
        if tool_name not in self.tool_implementations:
            raise ValueError(f"Tool {tool_name} not registered")
        
        implementation = self.tool_implementations[tool_name]
        return await implementation(**inputs)

    def _select_tool(
        self,
        context: Dict[str, Any],
        criteria: ToolSelectionCriteria,
        available_tools: List[Tool]
    ) -> tuple[str, float, List[str]]:
        """
        Implementation of tool selection logic - fallback if LLM is not available
        Returns: (selected_tool_name, confidence_score, reasoning_steps)
        """
        # Simple logic: Use weather_retriever first, then umbrella_decider
        task = context.get("task", "").lower()
        
        if "weather" in task or "forecast" in task:
            return (
                "weather_retriever",
                0.9,
                ["Task involves getting weather data", "Weather retriever tool is most appropriate"]
            )
        elif "umbrella" in task or "rain" in task:
            return (
                "umbrella_decider",
                0.9,
                ["Task involves deciding about umbrella need", "Umbrella decider tool is most appropriate"]
            )
        
        # Default to weather retriever as first step
        return (
            "weather_retriever",
            0.7,
            ["Default selection: weather data is needed first"]
        ) 