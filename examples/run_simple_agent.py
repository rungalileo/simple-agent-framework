import asyncio
from agent_framework.config import AgentConfiguration
from agent_framework.factory import AgentFactory
from examples.agents.simple_agent.agent import SimpleAgent

async def main():
    # Create configuration
    config = AgentConfiguration.from_env(
        required_keys=["openai"],
    )
    
    # Create factory with configuration
    factory = AgentFactory(config)
    
    # Create agent
    agent = factory.create_agent(
        agent_class=SimpleAgent,
        agent_id="simple_agent"
    )
    
    # Run agent
    result = await agent.run("What's the weather like in London?")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())