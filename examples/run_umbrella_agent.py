import asyncio
from agent_framework.models import VerbosityLevel
from agent_framework.utils.logging import AgentLogger
from examples.agents.umbrella_agent.agent import UmbrellaAgent

async def main():
    agent_logger = AgentLogger(agent_id="umbrella_agent")

    # Create agent with HIGH verbosity to see all logs
    agent = UmbrellaAgent(
        verbosity=VerbosityLevel.LOW,
        logger=agent_logger,
        metadata={"env": "example", "location": "Houston, TX"}
    )
    
    # Run the agent
    result = await agent.run("Houston, TX")
    print("\nFinal Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())