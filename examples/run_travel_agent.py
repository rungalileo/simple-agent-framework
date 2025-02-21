import asyncio
from agent_framework.config import AgentConfiguration
from agent_framework.factory import AgentFactory
from examples.agents.travel_agent.agent import TravelAgent
from agent_framework.models import VerbosityLevel

async def main():
    # Load configuration from environment with required keys
    config = AgentConfiguration.from_env(
        required_keys=["openai", "weather", "yelp", "ticketmaster"],
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
        agent_class=TravelAgent,
        agent_id="travel_agent"
    )
    
    await agent.run("I want to tour in New York, New York")

if __name__ == "__main__":
    asyncio.run(main())