from typing import Any, Dict, List
from agent_framework.agent import Agent
from agent_framework.models import Tool, ToolSelectionCriteria, VerbosityLevel
from agent_framework.llm.models import LLMConfig
from agent_framework.llm.openai_provider import OpenAIProvider
from agent_framework.config import load_config
from examples.agents.simple_agent.tools.text_analysis import TextAnalyzerTool
from examples.agents.simple_agent.tools.keyword_extraction import KeywordExtractorTool

class SimpleAgent(Agent):
    """Example implementation of the Agent class"""
    
    def __init__(
        self,
        *args,
        verbosity: VerbosityLevel = VerbosityLevel.LOW,
        **kwargs
    ):
        # Load configuration
        config = load_config()
        
        # Configure LLM provider
        llm_config = LLMConfig(
            model="gpt-4",
            temperature=0.7
        )
        llm_provider = OpenAIProvider(
            config=llm_config,
            api_key=config["openai_api_key"]
        )
        
        super().__init__(
            *args,
            llm_provider=llm_provider,
            verbosity=verbosity,
            **kwargs
        )
        
        # Register available tools
        self.register_tool(
            TextAnalyzerTool.get_tool_definition(),
            lambda text: TextAnalyzerTool.execute(text, self.llm_provider.config, self.llm_provider)
        )
        
        self.register_tool(
            KeywordExtractorTool.get_tool_definition(),
            lambda text: KeywordExtractorTool.execute(text, self.llm_provider.config, self.llm_provider)
        )

    def _select_tool(
        self,
        context: Dict[str, Any],
        criteria: ToolSelectionCriteria,
        available_tools: List[Tool]
    ) -> tuple[List[str], float, List[str]]:
        """Fallback method if LLM is not available"""
        return (
            [available_tools[0].name] if available_tools else [],
            0.5,
            ["Fallback selection: chose first available tool"]
        )

    async def _execute_task(self, task: str) -> str:
        """Execute the task using available tools"""
        # Log the initial step
        self.log_step(
            step_type="task_received",
            description=f"Received task: {task}",
            intermediate_state={"task": task}
        )
        
        # Let the base class handle planning and execution
        try:
            result = await super().run(task)
            
            # Log completion
            self.log_step(
                step_type="completion",
                description="Task completed successfully",
                intermediate_state={"final_result": result}
            )
            
            return result
            
        except Exception as e:
            # Log error
            self.log_step(
                step_type="error",
                description=f"Error executing task: {str(e)}",
                intermediate_state={"error": str(e)}
            )
            raise

    async def _execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of tool execution logic"""
        if tool_name not in self.tool_implementations:
            raise ValueError(f"Tool {tool_name} not registered")
        
        implementation = self.tool_implementations[tool_name]
        return await implementation(**inputs) 