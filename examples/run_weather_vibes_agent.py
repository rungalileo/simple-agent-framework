import asyncio
from agent_framework.config import AgentConfiguration
from agent_framework.factory import AgentFactory
from agents.weather_vibes_agent.agent import WeatherVibesAgent
from agent_framework.models import VerbosityLevel

async def main():
    # Load configuration from environment with required keys
    config = AgentConfiguration.from_env(
        required_keys=["openai", "weather", "youtube"],
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
        agent_class=WeatherVibesAgent,
        agent_id="weather-vibes-agent"
    )
    
    # Get location from command line or use default
    import sys
    location = sys.argv[1] if len(sys.argv) > 1 else "Seattle, WA"
    
    result = await agent.run(location)
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 