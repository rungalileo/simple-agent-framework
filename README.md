# Agent Framework

A flexible framework for building AI agents with tool-based capabilities.

## Overview

The Agent Framework provides a structured way to build AI agents that can use tools to accomplish tasks. Each agent:
- Has its own set of tools
- Manages its own state
- Has its own prompt templates
- Can be configured via environment variables or code

## Creating an Agent

Here's how to create a new agent:

1. Create the agent directory structure:
```
your_agent/
├── __init__.py
├── agent.py           # Agent implementation
├── templates/         # Agent-specific templates
│   └── planning.j2    # Planning prompt template
└── tools/            # Agent's tools
    ├── __init__.py
    ├── tool_one.py
    └── tool_two.py
```

2. Define your agent class:
```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from agent_framework.agent import Agent
from agent_framework.state import AgentState

class YourAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = AgentState()
        
        # Set up template environment
        template_dir = Path(__file__).parent / "templates"
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register agent-specific tools"""
        self.tool_registry.register(
            metadata=ToolOne.get_metadata(),
            implementation=ToolOne
        )
        # Register other tools...
```

3. Create your tools:
```python
from agent_framework.tools.base import BaseTool
from pydantic import BaseModel

class ToolOneInput(BaseModel):
    param1: str
    param2: int

class ToolOne(BaseTool):
    name = "tool_one"
    description = "Does something specific"
    tags = ["category1", "category2"]
    input_schema = ToolOneInput.model_json_schema()
    
    async def execute(self, param1: str, param2: int) -> Dict[str, Any]:
        # Tool implementation
        return {"result": "some result"}
```

## Using the Agent

1. Set up configuration:
```python
from agent_framework.config import AgentConfiguration
from agent_framework.models import VerbosityLevel

# Load config with required API keys
config = AgentConfiguration.from_env(
    required_keys=["openai"]  # List required API keys
)

# Override settings as needed
config = config.with_overrides(
    verbosity=VerbosityLevel.HIGH,
    enable_logging=False,
    metadata={"custom": "value"}
)
```

2. Create and run the agent:
```python
from agent_framework.factory import AgentFactory

# Create factory with config
factory = AgentFactory(config)

# Create agent instance
agent = factory.create_agent(
    agent_class=YourAgent,
    agent_id="unique_id"
)

# Run the agent
result = await agent.run("your task here")
```

## Environment Variables

Required variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- Other API keys as needed by your agent

Optional variables:
- `LLM_MODEL`: Model to use (default: "gpt-4")
- `LLM_TEMPERATURE`: Temperature setting (default: 0.1)
- `VERBOSITY`: Logging verbosity (low/medium/high)
- `ENABLE_LOGGING`: Enable/disable logging
- `ENVIRONMENT`: Environment name (development/production)

## Example

See the `examples/agents/umbrella_agent` for a complete example of an agent that determines if you need an umbrella based on weather data.

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
