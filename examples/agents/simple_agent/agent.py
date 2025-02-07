from typing import Any, Dict, List, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from agent_framework.agent import Agent
from agent_framework.state import AgentState
from agent_framework.models import VerbosityLevel, ToolSelectionHooks
from agent_framework.llm.models import LLMConfig
from agent_framework.llm.openai_provider import OpenAIProvider
from agent_framework.utils.logging import AgentLogger
from examples.agents.simple_agent.tools.text_analysis import TextAnalyzerTool
from examples.agents.simple_agent.tools.keyword_extraction import KeywordExtractorTool

class SimpleAgent(Agent):
    """A simple agent that demonstrates basic functionality"""
    
    def __init__(
        self,
        verbosity: VerbosityLevel = VerbosityLevel.LOW,
        logger: Optional[AgentLogger] = None,
        tool_selection_hooks: Optional[ToolSelectionHooks] = None,
        metadata: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[OpenAIProvider] = None,
    ):
        super().__init__(
            verbosity=verbosity,
            logger=logger,
            tool_selection_hooks=tool_selection_hooks,
            metadata=metadata,
            llm_provider=llm_provider
        )
        self.state = AgentState()
        
        # Set up template environment
        template_dir = Path(__file__).parent / "templates"
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Configure LLM provider if not provided
        if not self.llm_provider:
            llm_config = LLMConfig(
                model="gpt-4",
                temperature=0.7
            )
            self.llm_provider = OpenAIProvider(config=llm_config)
        
        # Register tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all tools with the registry"""
        # Text analyzer
        self.tool_registry.register(
            metadata=TextAnalyzerTool.get_metadata(),
            implementation=TextAnalyzerTool
        )
        
        # Keyword extractor
        self.tool_registry.register(
            metadata=KeywordExtractorTool.get_metadata(),
            implementation=KeywordExtractorTool
        )

    async def _format_result(self, task: str, results: List[tuple[str, Dict[str, Any]]]) -> str:
        """Format the final result from tool executions"""
        # Simple agent just returns the raw results
        return str(results) 