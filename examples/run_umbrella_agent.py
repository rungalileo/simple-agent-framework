import asyncio
from agent_framework.config import AgentConfiguration
from agent_framework.factory import AgentFactory
from examples.agents.umbrella_agent.agent import UmbrellaAgent
from agent_framework.models import VerbosityLevel

async def main():
    # Load configuration from environment with required keys
    config = AgentConfiguration.from_env(
        required_keys=["openai", "weather"],
    )
    
    # Override specific settings
    config = config.with_overrides(
        verbosity=VerbosityLevel.HIGH,
        enable_logging=False,
        metadata={"location": "Seattle, WA"}
    )
    
    # Create factory and agent
    factory = AgentFactory(config)
    agent = factory.create_agent(
        agent_class=UmbrellaAgent,
        agent_id="umbrella-agent"
    )
    
    await agent.run("Seattle, WA")

if __name__ == "__main__":
    asyncio.run(main())