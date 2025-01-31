from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class AgentMetadata(BaseModel):
    """Metadata associated with an agent"""
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    capabilities: List[str] = Field(default_factory=list)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)

class Tool(BaseModel):
    """Definition of a tool available to agents"""
    name: str = Field(
        description="Unique identifier for the tool"
    )
    description: str = Field(
        description="Human-readable description of what the tool does and when to use it"
    )
    input_schema: Dict[str, Any] = Field(
        description="JSON Schema defining the expected input parameters and their types"
    )
    output_schema: Dict[str, Any] = Field(
        description="JSON Schema defining the structure and types of the tool's output"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Categories or labels that describe the tool's capabilities and use cases"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional limitations or requirements for using the tool"
    )

class ToolSelectionCriteria(BaseModel):
    """Criteria used for selecting a tool"""
    required_tags: List[str] = Field(
        default_factory=list,
        description="Tags that a tool must have to be considered"
    )
    preferred_tags: List[str] = Field(
        default_factory=list,
        description="Tags that are desired but not required in a tool"
    )
    context_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific contextual requirements that influence tool selection"
    )
    custom_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional rules or criteria for tool selection"
    )

class ToolSelectionReasoning(BaseModel):
    """Record of the reasoning process for tool selection"""
    context: Dict[str, Any] = Field(
        description="Current context and state when the tool selection was made"
    )
    considered_tools: List[str] = Field(
        description="Names of all tools that were evaluated for selection"
    )
    selection_criteria: ToolSelectionCriteria = Field(
        description="Criteria used to evaluate and select the tool"
    )
    reasoning_steps: List[str] = Field(
        description="Detailed steps of the decision-making process"
    )
    selected_tool: str = Field(
        description="Name of the tool that was ultimately chosen"
    )
    confidence_score: float = Field(
        description="Confidence level in the tool selection (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

class VerbosityLevel(str, Enum):
    """Verbosity level for agent logging"""
    NONE = "none"  # No logging
    LOW = "low"    # Basic logging
    HIGH = "high"  # Detailed logging including tool selection and reasoning

class ToolCall(BaseModel):
    """Record of a tool invocation"""
    tool_name: str = Field(
        description="Name of the tool that was called"
    )
    inputs: Dict[str, Any] = Field(
        description="Parameters passed to the tool during execution"
    )
    outputs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results returned by the tool execution"
    )
    selection_reasoning: Optional[ToolSelectionReasoning] = Field(
        default=None,
        description="Reasoning process that led to selecting this tool"
    )
    execution_reasoning: str = Field(
        description="Explanation of why this tool was executed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the tool was called"
    )
    success: bool = Field(
        default=True,
        description="Whether the tool execution completed successfully"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the tool execution failed"
    )

class TaskAnalysis(BaseModel):
    """Analysis of a task using chain of thought reasoning"""
    input_analysis: str = Field(
        description="Analysis of the input, identifying key requirements and constraints"
    )
    available_tools: List[str] = Field(
        description="List of tools available for the task"
    )
    tool_capabilities: Dict[str, List[str]] = Field(
        description="Mapping of tools to their key capabilities"
    )
    execution_plan: List[Dict[str, Any]] = Field(
        description="Ordered list of steps to execute, each with tool and reasoning"
    )
    requirements_coverage: Dict[str, List[str]] = Field(
        description="How the identified requirements are covered by the planned steps"
    )
    chain_of_thought: List[str] = Field(
        description="Chain of thought reasoning that led to this plan"
    )

class ExecutionStep(BaseModel):
    """Record of a single step in the agent's execution"""
    step_type: str = Field(
        description="Category or type of execution step (e.g., 'task_received', 'processing', 'completion')"
    )
    description: str = Field(
        description="Human-readable description of what happened in this step"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the step occurred"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Tools that were called during this step"
    )
    intermediate_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="State or context information captured during this step"
    )

class TaskExecution(BaseModel):
    """Complete record of a task execution"""
    task_id: str = Field(
        description="Unique identifier for this task execution"
    )
    agent_id: str = Field(
        description="Identifier of the agent executing the task"
    )
    input: str = Field(
        description="Original input or request given to the agent"
    )
    steps: List[ExecutionStep] = Field(
        default_factory=list,
        description="Sequence of steps taken during task execution"
    )
    output: Optional[str] = Field(
        default=None,
        description="Final result or response from the task execution"
    )
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when task execution began"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when task execution completed"
    )
    status: str = Field(
        default="in_progress",
        description="Current status of the task (e.g., 'in_progress', 'completed', 'failed')"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the task execution failed"
    )