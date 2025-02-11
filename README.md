# Agent Framework

A flexible framework for building AI agents with tool-based capabilities.

## Overview

The Agent Framework provides a structured way to build AI agents that can use tools to accomplish tasks. Each agent:
- Has its own set of tools
- Manages its own state
- Has its own prompt templates
- Can be configured via environment variables or code
- Supports advanced logging with GalileoLogger

## Creating an Agent

Here's how to create a new agent:

1. Create the agent directory structure:
```
your_agent/
├── __init__.py
├── agent.py           # Agent implementation
├── templates/         # Agent-specific templates
│   └── planning.j2    # Planning prompt template
├── logging/          # Agent-specific logging
│   ├── __init__.py
│   └── GalileoAgentLogger.py
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
from .logging.GalileoAgentLogger import GalileoAgentLogger

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
        
        # Initialize Galileo logger
        self.logger = GalileoAgentLogger(agent_id=self.agent_id)
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register agent-specific tools"""
        # Register tools
        self.tool_registry.register(
            metadata=ToolOne.get_metadata(),
            implementation=ToolOne
        )
        # Register other tools...
        
        # Set up logger with tools
        self._setup_logger(logger=self.logger)
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

4. Set up the GalileoLogger:
```python
from galileo_observe import ObserveWorkflows
from agent_framework.utils.logging import AgentLogger
from agent_framework.utils.hooks import ToolHooks, ToolSelectionHooks
from agent_framework.models import ToolContext

class GalileoAgentLogger(AgentLogger):
    """Main logger interface for the agent"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.observe_logger = ObserveWorkflows(project_name="your-agent-name")
        
    async def on_agent_planning(self, planning_prompt: str) -> None:
        # Initialize workflow
        self.workflow = self.observe_logger.add_agent_workflow(
            input=planning_prompt,
            name="your_agent",
            metadata={"agent_id": self.agent_id}
        )

    async def on_agent_done(self, result: Any, message_history: Optional[List[Any]] = None) -> None:
        # Log final result and upload
        await self.workflow.conclude(output={"result": result})
        self.observe_logger.upload_workflows()

    def get_tool_hooks(self) -> ToolHooks:
        """Create tool execution hooks"""
        class Hooks(ToolHooks):
            async def before_execution(self, context: ToolContext) -> None:
                pass

            async def after_execution(self, context: ToolContext, result: Any, 
                                   error: Optional[Exception] = None) -> None:
                if not error:
                    # Log successful tool execution
                    pass

        return Hooks()

    def get_tool_selection_hooks(self) -> ToolSelectionHooks:
        """Create tool selection hooks"""
        class Hooks(ToolSelectionHooks):
            async def after_selection(self, context: ToolContext, selected_tool: str,
                                   confidence: float, reasoning: List[str]) -> None:
                # Log tool selection
                pass

        return Hooks()
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
    enable_logging=True,  # Enable Galileo logging
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
- `GALILEO_API_KEY`: Your Galileo API key (for logging)
- Other API keys as needed by your agent

Optional variables:
- `LLM_MODEL`: Model to use (default: "gpt-4")
- `LLM_TEMPERATURE`: Temperature setting (default: 0.1)
- `VERBOSITY`: Logging verbosity (low/medium/high)
- `ENABLE_LOGGING`: Enable/disable logging
- `ENVIRONMENT`: Environment name (development/production)

## Example

See the `examples/agents/umbrella_agent` for a complete example of an agent that determines if you need an umbrella based on weather data, including Galileo logging integration.

## Key Components

- **Agent**: Base class for all agents (`agent_framework/agent.py`)
- **Tools**: Reusable functions that agents can leverage
- **LLM Providers**: Interfaces to language models (currently supports OpenAI)
- **Task Execution**: Structured workflow for handling tasks and tool selection
- **Logging**: Advanced logging capabilities with GalileoLogger

## Best Practices

1. **Tool Design**:
   - Keep tools focused and single-purpose
   - Provide clear descriptions and parameter documentation
   - Handle errors gracefully

2. **Agent Implementation**:
   - Override `_select_tool` for custom tool selection logic
   - Use appropriate temperature settings for your use case
   - Implement proper error handling and logging
   - Use GalileoLogger for detailed execution tracking

3. **Logging**:
   - Initialize GalileoLogger in your agent
   - Implement appropriate hooks for tool execution and selection
   - Use workflow tracking for complex agent tasks
   - Log important state changes and decisions

4. **Security**:
   - Never hardcode API keys
   - Use environment variables or config files for sensitive data
   - Validate and sanitize inputs to tools
