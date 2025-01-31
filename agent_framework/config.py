from functools import lru_cache
import os
from typing import Optional
from dotenv import load_dotenv

class EnvironmentError(Exception):
    """Raised when required environment variables are missing"""
    pass

@lru_cache()
def load_config():
    """Load configuration from environment variables"""
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your .env file"
        )
    
    weather_api_key = os.getenv("WEATHER_API_KEY")
    if not weather_api_key:
        raise EnvironmentError(
            "WEATHER_API_KEY environment variable is required. "
            "Please set it in your .env file"
        )
    
    return {
        "openai_api_key": openai_api_key,
        "openai_org": os.getenv("OPENAI_ORGANIZATION"),  # Optional
        "weather_api_key": weather_api_key
    } 