from abc import ABC, abstractmethod
from typing import Any, Dict, ClassVar, Type
from ..models import ToolMetadata

class BaseTool(ABC):
    """Base class for all tools"""
    
    # Tool metadata as class variables
    metadata: ClassVar[Type[ToolMetadata]]
    
    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """Get tool metadata for planning"""
        # Create an instance of the metadata class
        return cls.metadata()  # This will use the default values defined in the metadata class
    
    @abstractmethod
    async def execute(self, **inputs: Any) -> Dict[str, Any]:
        """Execute the tool with given inputs"""
        pass 