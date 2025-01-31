import asyncio
from examples.agents.simple_agent.agent import SimpleAgent

async def main():
    # Create agent instance
    agent = SimpleAgent()
    
    # Run a sample task
    result = await agent.run("Analyze this text for complexity")
    print(f"Result: {result}")
    
    # Print execution steps
    print("\nExecution Steps:")
    for step in agent.current_task.steps:
        print(f"\nStep Type: {step.step_type}")
        print(f"Description: {step.description}")
        if step.tool_calls:
            print("Tool Calls:")
            for tool_call in step.tool_calls:
                print(f"  Tool: {tool_call.tool_name}")
                print(f"  Inputs: {tool_call.inputs}")
                print(f"  Outputs: {tool_call.outputs}")

if __name__ == "__main__":
    asyncio.run(main()) 