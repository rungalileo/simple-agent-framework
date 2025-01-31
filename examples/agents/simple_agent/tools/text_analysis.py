from typing import Any, Dict
from agent_framework.models import Tool
from agent_framework.llm.models import LLMMessage, LLMConfig
from agent_framework.llm.tool_models import TextAnalysis

class TextAnalyzerTool:
    """Text analysis tool implementation"""
    
    @staticmethod
    def get_tool_definition() -> Tool:
        return Tool(
            name="text_analyzer",
            description="Analyzes text for complexity, readability, topics, and key points",
            input_schema={"text": "string"},
            output_schema=TextAnalysis.model_json_schema(),
            tags=["text", "analysis", "complexity", "topics"]
        )

    @staticmethod
    async def execute(text: str, llm_config: LLMConfig, llm_provider: Any) -> Dict[str, Any]:
        """Execute the text analysis"""
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are an advanced text analysis system. Analyze the provided text "
                    "for complexity, readability, main topics, and key points. Provide a "
                    "detailed analysis with metrics and insights."
                    "\n\nYour analysis must include:\n"
                    "1. A complexity score (0-1)\n"
                    "2. Readability level (e.g., Basic, Intermediate, Advanced)\n"
                    "3. Main topics identified\n"
                    "4. Key points from the text\n"
                    "5. A summary of the analysis\n"
                    "6. Language metrics including:\n"
                    "   - Sentence count\n"
                    "   - Average sentence length\n"
                    "   - Vocabulary richness (0-1)\n"
                    "   - Any other relevant metrics\n\n"
                    "Your response must exactly match this JSON structure:\n"
                    "{\n"
                    '  "complexity_score": 0.75,\n'
                    '  "readability_level": "Advanced",\n'
                    '  "main_topics": ["topic1", "topic2"],\n'
                    '  "key_points": ["point1", "point2", "point3"],\n'
                    '  "analysis_summary": "Summary of the analysis",\n'
                    '  "language_metrics": {\n'
                    '    "sentence_count": 15,\n'
                    '    "average_sentence_length": 20,\n'
                    '    "vocabulary_richness": 0.8,\n'
                    '    "additional_metric": "value"\n'
                    '  }\n'
                    "}"
                )
            ),
            LLMMessage(
                role="user",
                content=f"Analyze this text:\n\n{text}"
            )
        ]
        
        analysis = await llm_provider.generate_structured(
            messages,
            TextAnalysis,
            llm_config
        )
        
        return analysis.model_dump() 