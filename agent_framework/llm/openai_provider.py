from typing import Any, Dict, List, Optional, AsyncGenerator, Type, TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel

from .base import LLMProvider
from .models import LLMMessage, LLMResponse, LLMConfig

T = TypeVar('T', bound=BaseModel)

class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider"""
    
    def __init__(
        self,
        config: LLMConfig,
        api_key: str,
        organization: Optional[str] = None
    ):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization
        )

    def _prepare_messages(
        self,
        messages: List[LLMMessage]
    ) -> List[Dict[str, Any]]:
        """Convert internal message format to OpenAI format"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]

    def _prepare_config(self, config: Optional[LLMConfig] = None) -> Dict[str, Any]:
        """Prepare configuration for OpenAI API"""
        cfg = config or self.config
        return {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "frequency_penalty": cfg.frequency_penalty,
            "presence_penalty": cfg.presence_penalty,
            "stop": cfg.stop,
            **cfg.custom_settings
        }

    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate a response using OpenAI"""
        openai_messages = self._prepare_messages(messages)
        api_config = self._prepare_config(config)
        
        response = await self.client.chat.completions.create(
            messages=openai_messages,
            **api_config
        )
        
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            raw_response=response.model_dump(),
            finish_reason=choice.finish_reason,
            usage=response.usage.model_dump() if response.usage else None
        )

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response using OpenAI"""
        openai_messages = self._prepare_messages(messages)
        api_config = self._prepare_config(config)
        
        stream = await self.client.chat.completions.create(
            messages=openai_messages,
            stream=True,
            **api_config
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield LLMResponse(
                    content=chunk.choices[0].delta.content,
                    raw_response=chunk.model_dump(),
                    finish_reason=chunk.choices[0].finish_reason
                ) 

    async def generate_structured(
        self,
        messages: List[LLMMessage],
        output_model: Type[T],
        config: Optional[LLMConfig] = None
    ) -> T:
        """Generate a response with structured output using function calling"""
        openai_messages = self._prepare_messages(messages)
        api_config = self._prepare_config(config)
        
        # Create function definition from Pydantic model
        schema = output_model.model_json_schema()
        function_def = {
            "name": "output_structured_data",
            "description": f"Output data in {output_model.__name__} format",
            "parameters": schema
        }
        
        response = await self.client.chat.completions.create(
            messages=openai_messages,
            functions=[function_def],
            function_call={"name": "output_structured_data"},
            **api_config
        )
        
        try:
            function_args = response.choices[0].message.function_call.arguments
            return output_model.model_validate_json(function_args)
        except Exception as e:
            raise ValueError(f"Failed to parse structured output: {e}") 