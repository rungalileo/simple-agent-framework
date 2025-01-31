from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class ToolMetadata(BaseModel):
    """Base schema for tool metadata"""
    name: str = Field(description="Unique identifier for the tool")
    description: str = Field(description="Human-readable description of what the tool does")
    tags: List[str] = Field(description="Categories/capabilities of the tool")
    input_schema: Dict[str, Any] = Field(description="JSON schema for tool inputs")
    output_schema: Dict[str, Any] = Field(description="JSON schema for tool outputs")
    examples: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Example uses of the tool"
    )

class AgentMetadata(BaseModel):
    """Base schema for agent metadata"""
    name: str = Field(description="Name of the agent")
    description: str = Field(description="What the agent does")
    capabilities: List[str] = Field(description="High-level capabilities")
    tools: List[ToolMetadata] = Field(description="Tools available to this agent")
    model_config = ConfigDict(arbitrary_types_allowed=True) 