import asyncio
from agent_framework.models import VerbosityLevel
from examples.agents.simple_agent.agent import SimpleAgent

async def main():
    # Create agent instance with high verbosity
    agent = SimpleAgent(verbosity=VerbosityLevel.HIGH)
    
    # Run a sample task    
    result = await agent.run("Analyze this text for complexity and extract keywords like 'text', 'analysis', 'complexity'")
    print(f"\nFinal Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())