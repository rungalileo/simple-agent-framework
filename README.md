# Simple Agent Framework

A lightweight framework for building AI agents with LLM capabilities.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

1. Create a new agent by subclassing the base `Agent` class:

```python
from agent_framework.agent import Agent
from agent_framework.llm.models import LLMConfig
from agent_framework.llm.openai_provider import OpenAIProvider

class MyAgent(Agent):
    def __init__(self, *args, **kwargs):
        # Configure LLM provider
        llm_config = LLMConfig(
            model="gpt-4",
            temperature=0.7
        )
        llm_provider = OpenAIProvider(
            config=llm_config,
            api_key="your_api_key"
        )
        
        super().__init__(
            *args,
            llm_provider=llm_provider,
            **kwargs
        )
```

2. Register tools that your agent can use:

```python
from agent_framework.models import Tool

# Define a tool
my_tool = Tool(
    name="example_tool",
    description="A tool that does something",
    parameters={
        "input_text": "The text to process"
    }
)

# Implement the tool's functionality
async def tool_implementation(text):
    # Your tool logic here
    return {"result": processed_text}

# Register the tool with your agent
agent.register_tool(
    my_tool,
    tool_implementation
)
```

3. Run your agent:

```python
async def main():
    agent = MyAgent()
    result = await agent.run("Perform a task using the available tools")
    print(f"Result: {result}")
```

## Examples

The framework includes two example implementations:

1. **SimpleAgent**: A basic agent that demonstrates core functionality with text analysis tools.
   - See `examples/agents/simple_agent/agent.py`
   - Run with `python examples/run_simple_agent.py`

2. **UmbrellaAgent**: A more complex agent that can coordinate multiple sub-agents.
   - See `examples/agents/umbrella_agent/agent.py`
   - Run with `python examples/run_umbrella_agent.py`

## Key Components

- **Agent**: Base class for all agents (`agent_framework/agent.py`)
- **Tools**: Reusable functions that agents can leverage
- **LLM Providers**: Interfaces to language models (currently supports OpenAI)
- **Task Execution**: Structured workflow for handling tasks and tool selection

## Configuration

Create a `config.yaml` file in your project root:

```yaml
openai_api_key: "your_api_key_here"
```

## Best Practices

1. **Tool Design**:
   - Keep tools focused and single-purpose
   - Provide clear descriptions and parameter documentation
   - Handle errors gracefully

2. **Agent Implementation**:
   - Override `_select_tool` for custom tool selection logic
   - Use appropriate temperature settings for your use case
   - Implement proper error handling and logging

3. **Security**:
   - Never hardcode API keys
   - Use environment variables or config files for sensitive data
   - Validate and sanitize inputs to tools
