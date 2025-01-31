from typing import Optional, Type
from .config import AgentConfiguration
from .llm.base import LLMProvider
from .llm.openai_provider import OpenAIProvider
from .utils.logging import AgentLogger
from .agent import Agent

class AgentFactory:
    """Factory for creating agents with proper dependency injection"""
    
    def __init__(self, config: AgentConfiguration):
        self.config = config
        self._llm_provider: Optional[LLMProvider] = None
        self._logger: Optional[AgentLogger] = None
    
    def get_llm_provider(self) -> LLMProvider:
        """Get or create LLM provider"""
        if not self._llm_provider:
            if "openai" in self.config.api_keys:
                self._llm_provider = OpenAIProvider(
                    config=self.config.llm_config,
                    api_key=self.config.api_keys["openai"]
                )
            else:
                raise ValueError("No LLM provider configured")
        return self._llm_provider
    
    def get_logger(self, agent_id: str) -> Optional[AgentLogger]:
        """Get logger if enabled"""
        if self.config.enable_logging:
            return AgentLogger(agent_id)
        return None
    
    def create_agent(
        self,
        agent_class: Type[Agent],
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Agent:
        """Create an agent instance with proper dependencies"""
        # Create core dependencies
        llm_provider = self.get_llm_provider()
        logger = self.get_logger(agent_id) if agent_id else None
        
        # Create agent with injected dependencies
        agent = agent_class(
            llm_provider=llm_provider,
            logger=logger,
            verbosity=self.config.verbosity,
            metadata=self.config.metadata,
            **kwargs
        )
        
        return agent 