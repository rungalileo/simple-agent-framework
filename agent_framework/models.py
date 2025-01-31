from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class AgentMetadata(BaseModel):
    """Metadata associated with an agent"""
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    capabilities: List[str] = Field(default_factory=list)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)

class Tool(BaseModel):
    """Definition of a tool available to agents"""
    name: str
    description: str
    input_schema: Dict[str, Any]  # Expected input parameters
    output_schema: Dict[str, Any]  # Expected output format
    tags: List[str] = Field(default_factory=list)  # For categorization
    constraints: Optional[Dict[str, Any]] = None  # Usage constraints/requirements

class ToolSelectionCriteria(BaseModel):
    """Criteria used for selecting a tool"""
    required_tags: List[str] = Field(default_factory=list)
    preferred_tags: List[str] = Field(default_factory=list)
    context_requirements: Dict[str, Any] = Field(default_factory=dict)
    custom_rules: Dict[str, Any] = Field(default_factory=dict)

class ToolSelectionReasoning(BaseModel):
    """Record of the reasoning process for tool selection"""
    context: Dict[str, Any]  # Current context when selection was made
    considered_tools: List[str]  # Tools that were considered
    selection_criteria: ToolSelectionCriteria
    reasoning_steps: List[str]  # Step-by-step reasoning process
    selected_tool: str
    confidence_score: float  # 0-1 score of selection confidence

class ToolCall(BaseModel):
    """Record of a tool invocation"""
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    selection_reasoning: Optional[ToolSelectionReasoning] = None  # New field
    execution_reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error: Optional[str] = None

class ExecutionStep(BaseModel):
    """Record of a single step in the agent's execution"""
    step_type: str
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    intermediate_state: Optional[Dict[str, Any]] = None

class TaskExecution(BaseModel):
    """Complete record of a task execution"""
    task_id: str
    agent_id: str
    input: str
    steps: List[ExecutionStep] = Field(default_factory=list)
    output: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "in_progress" 