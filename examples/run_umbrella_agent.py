import asyncio
from examples.agents.umbrella_agent.agent import UmbrellaAgent

async def main():
    agent = UmbrellaAgent()
    result = await agent.run("Houston, TX")
    print(result)

    result = await agent.run("Seattle, WA")
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 