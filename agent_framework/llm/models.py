from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class LLMMessage(BaseModel):
    """Message format for LLM interactions"""
    role: str
    content: str
    name: Optional[str] = None

class LLMResponse(BaseModel):
    """Structured response from LLM"""
    content: str
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

class LLMConfig(BaseModel):
    """Configuration for LLM provider"""
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    custom_settings: Dict[str, Any] = Field(default_factory=dict)

class ToolSelectionOutput(BaseModel):
    """Structured output format for tool selection"""
    selected_tool: str
    confidence: float = Field(ge=0.0, le=1.0)
    task_analysis: str = Field(description="Analysis of the task requirements")
    reasoning_steps: List[str] = Field(min_items=1)

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        """Get JSON schema with example"""
        schema = super().model_json_schema()
        schema["examples"] = [{
            "selected_tool": "text_analyzer",
            "confidence": 0.9,
            "task_analysis": "The task requires deep analysis of text complexity and structure",
            "reasoning_steps": [
                "Task involves understanding text complexity",
                "Text analyzer provides detailed analysis capabilities",
                "Other tools don't provide complexity metrics"
            ]
        }]
        return schema 