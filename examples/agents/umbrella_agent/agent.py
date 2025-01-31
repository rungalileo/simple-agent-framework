from typing import Any, Dict, List
from uuid import uuid4
from datetime import datetime
from agent_framework.agent import Agent
from agent_framework.models import Tool, ToolSelectionCriteria, VerbosityLevel, TaskExecution
from agent_framework.llm.models import LLMConfig, LLMMessage
from agent_framework.llm.openai_provider import OpenAIProvider
from agent_framework.config import load_config
from agent_framework.utils.formatting import display_tool_result, display_final_result, display_error
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

    def _create_planning_prompt(self, task: str) -> List[LLMMessage]:
        """Create a custom planning prompt for the umbrella agent"""
        tools_description = "\n".join([
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Tags: {', '.join(tool.tags)}\n"
            f"Input Schema: {tool.input_schema}\n"
            f"Output Schema: {tool.output_schema}\n"
            for tool in self.tools.values()
        ])

        system_prompt = (
            "You are an intelligent weather analysis system that helps users decide if they need an umbrella.\n\n"
            "You MUST follow this exact sequence:\n"
            "1. First, get the weather data using the weather_retriever tool\n"
            "2. Then, analyze the weather data using the umbrella_decider tool\n\n"
            "You MUST provide a complete response with ALL of the following components:\n\n"
            "1. input_analysis: Analyze the location and what weather information we need\n"
            "2. available_tools: List the weather_retriever and umbrella_decider tools\n"
            "3. tool_capabilities: Map each tool to its capabilities\n"
            "4. execution_plan: MUST be exactly 2 steps in this order:\n"
            "   - First step: weather_retriever to get weather data\n"
            "   - Second step: umbrella_decider to make the decision\n"
            "5. requirements_coverage: How the tools cover our needs\n"
            "6. chain_of_thought: Your reasoning about the weather analysis\n\n"
            f"Available Tools:\n{tools_description}\n\n"
            "Your response MUST be a JSON object with this EXACT structure:\n"
            "{\n"
            '  "input_analysis": "Analysis of the location and weather requirements",\n'
            '  "available_tools": ["weather_retriever", "umbrella_decider"],\n'
            '  "tool_capabilities": {\n'
            '    "weather_retriever": ["get weather data", "check precipitation"],\n'
            '    "umbrella_decider": ["analyze weather data", "make umbrella decision"]\n'
            "  },\n"
            '  "execution_plan": [\n'
            '    {"tool": "weather_retriever", "reasoning": "First, get weather data"},\n'
            '    {"tool": "umbrella_decider", "reasoning": "Then, decide if umbrella needed"}\n'
            "  ],\n"
            '  "requirements_coverage": {\n'
            '    "weather_data": ["weather_retriever"],\n'
            '    "umbrella_decision": ["umbrella_decider"]\n'
            "  },\n"
            '  "chain_of_thought": [\n'
            '    "First, we need current weather data",\n'
            '    "Then we can analyze it for umbrella needs"\n'
            "  ]\n"
            "}\n\n"
            "Ensure ALL fields are present and properly formatted. Missing fields will cause errors."
        )

        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Location: {task}\n\nAnalyze this location and create a complete weather analysis plan with ALL required fields."
            )
        ]

    async def run(self, task: str) -> str:
        """Execute a task and return the result"""
        self.current_task = TaskExecution(
            task_id=str(uuid4()),
            agent_id=self.agent_id,
            input=task,
            start_time=datetime.now(),
            steps=[]
        )
        
        try:
            # Create a plan using chain of thought reasoning
            plan = await self.plan_task(task)
            
            # Execute each step in the plan
            results = []
            for step in plan.execution_plan:
                tool_name = step["tool"]
                if tool_name not in self.tools:
                    raise ValueError(f"Tool {tool_name} not found")
                
                # For weather_retriever, use location as input
                if tool_name == "weather_retriever":
                    result = await self.call_tool(
                        tool_name=tool_name,
                        inputs={"location": task},
                        execution_reasoning=step["reasoning"],
                        context={"task": task, "plan": plan}
                    )
                    self._weather_data = result  # Store for umbrella_decider
                
                # For umbrella_decider, use stored weather data
                elif tool_name == "umbrella_decider":
                    if not hasattr(self, '_weather_data'):
                        raise ValueError("Weather data not available for umbrella decision")
                    result = await self.call_tool(
                        tool_name=tool_name,
                        inputs={"weather_data": self._weather_data},
                        execution_reasoning=step["reasoning"],
                        context={"task": task, "plan": plan}
                    )
                
                if self.verbosity == VerbosityLevel.HIGH:
                    display_tool_result(tool_name, result)
                
                results.append((tool_name, result))
            
            # Create final result with weather details
            weather_data = self._weather_data
            umbrella_needed = results[1][1]  # Result from umbrella_decider
            
            result = "You need an umbrella today!" if umbrella_needed else "No umbrella needed today!"
            result += f"\n\nWeather details for {weather_data['location']}:"
            result += f"\n- Temperature: {weather_data.get('temperature', 'N/A')}°C"
            result += f"\n- Condition: {weather_data.get('weather_condition', 'N/A')}"
            result += f"\n- Chance of rain: {weather_data.get('precipitation_chance', 'N/A')}%"
            
            self.current_task.output = result
            
            if self.verbosity == VerbosityLevel.HIGH:
                display_final_result(result)
            
            return result
            
        except Exception as e:
            self.current_task.error = str(e)
            self.current_task.status = "failed"
            
            if self.verbosity == VerbosityLevel.HIGH:
                display_error(str(e))
            
            raise
        finally:
            self.current_task.end_time = datetime.now()
            if self.current_task.status == "in_progress":
                self.current_task.status = "completed"

    async def _execute_task(self, task: str) -> str:
        """This is now handled by the run method"""
        return await self.run(task)

    def _select_tool(
        self,
        task: str,
        tools: List[Tool],
        previous_tools: List[str] = None,
        previous_results: List[Dict[str, Any]] = None,
        previous_errors: List[str] = None
    ) -> ToolSelectionCriteria:
        """Select the next tool to use based on the task and previous results"""
        if not previous_tools:
            # First tool should always be weather_retriever
            return ToolSelectionCriteria(
                "weather_retriever",
                1.0,
                ["Initial step: Get weather data"]
            )
        elif len(previous_tools) == 1 and previous_tools[0] == "weather_retriever":
            # Second tool should always be umbrella_decider
            return ToolSelectionCriteria(
                "umbrella_decider",
                1.0,
                ["Second step: Decide if umbrella is needed"]
            )
        else:
            # Default to weather_retriever if we don't know what to do
            return ToolSelectionCriteria(
                "weather_retriever",
                0.7,
                ["Default selection: weather data is needed first"]
            ) 

    async def _map_inputs_to_tool(self, tool_name: str, task: str, input_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map task input to tool-specific inputs"""
        if tool_name == "umbrella_decider":
            # Get the weather data from the previous step
            if not hasattr(self, '_weather_data'):
                raise ValueError("Weather data not available for umbrella decision")
            return {"weather_data": self._weather_data}
        
        # For other tools, use default mapping
        return await super()._map_inputs_to_tool(tool_name, task, input_mapping)