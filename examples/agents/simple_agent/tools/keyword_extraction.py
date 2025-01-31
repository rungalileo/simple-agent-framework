from typing import Any, Dict
from agent_framework.models import Tool
from agent_framework.llm.models import LLMMessage, LLMConfig
from agent_framework.llm.tool_models import KeywordExtraction

class KeywordExtractorTool:
    """Keyword extraction tool implementation"""
    
    @staticmethod
    def get_tool_definition() -> Tool:
        return Tool(
            name="keyword_extractor",
            description="Extracts and categorizes keywords with importance scoring",
            input_schema={"text": "string"},
            output_schema=KeywordExtraction.model_json_schema(),
            tags=["text", "keywords", "extraction", "categorization"]
        )

    @staticmethod
    async def execute(text: str, llm_config: LLMConfig, llm_provider: Any) -> Dict[str, Any]:
        """Execute the keyword extraction"""
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are an advanced keyword extraction system. Extract relevant keywords "
                    "from the provided text, score their importance, and categorize them. "
                    "Consider the context and domain of the text in your analysis."
                    "\n\nYour output must include:\n"
                    "1. A list of keywords\n"
                    "2. Importance scores (0-1) for each keyword\n"
                    "3. Categories with their associated keywords\n"
                    "4. Overall extraction confidence (0-1)\n"
                    "5. Context relevance description\n\n"
                    "Your response must exactly match this JSON structure:\n"
                    "{\n"
                    '  "keywords": ["keyword1", "keyword2", "keyword3"],\n'
                    '  "importance_scores": {\n'
                    '    "keyword1": 0.9,\n'
                    '    "keyword2": 0.8,\n'
                    '    "keyword3": 0.7\n'
                    '  },\n'
                    '  "categories": {\n'
                    '    "category1": ["keyword1", "keyword2"],\n'
                    '    "category2": ["keyword3"]\n'
                    '  },\n'
                    '  "extraction_confidence": 0.85,\n'
                    '  "context_relevance": "Description of relevance to context"\n'
                    "}"
                )
            ),
            LLMMessage(
                role="user",
                content=f"Extract keywords from this text:\n\n{text}"
            )
        ]
        
        extraction = await llm_provider.generate_structured(
            messages,
            KeywordExtraction,
            llm_config
        )
        
        return extraction.model_dump() 