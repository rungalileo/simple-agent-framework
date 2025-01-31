import asyncio
from agent_framework.models import VerbosityLevel

from examples.agents.umbrella_agent.agent import UmbrellaAgent

async def main():
    agent = UmbrellaAgent(verbosity=VerbosityLevel.HIGH)
    result = await agent.run("Houston, TX")
    print(result)

    result = await agent.run("Seattle, WA")
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 