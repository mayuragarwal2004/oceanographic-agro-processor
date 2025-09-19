"""
Abstract LLM Connector Architecture
Allows easy switching between different LLM providers (GROQ, OpenAI, Anthropic, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass
from enum import Enum

class LLMProvider(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

@dataclass
class LLMMessage:
    role: str  # "system", "user", "assistant"
    content: str

@dataclass
class LLMResponse:
    content: str
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseLLMConnector(ABC):
    """Abstract base class for LLM connectors"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    async def generate(
        self, 
        messages: List[LLMMessage], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate the API connection"""
        pass

class LLMConnectorFactory:
    """Factory for creating LLM connectors"""
    
    @staticmethod
    def create_connector(
        provider: LLMProvider, 
        api_key: str, 
        model: str, 
        **kwargs
    ) -> BaseLLMConnector:
        """Create an LLM connector based on provider"""
        
        if provider == LLMProvider.GROQ:
            from .groq_connector import GroqConnector
            return GroqConnector(api_key, model, **kwargs)
        elif provider == LLMProvider.OPENAI:
            from .openai_connector import OpenAIConnector
            return OpenAIConnector(api_key, model, **kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            from .anthropic_connector import AnthropicConnector
            return AnthropicConnector(api_key, model, **kwargs)
        elif provider == LLMProvider.GEMINI:
            from .gemini_connector import GeminiConnector
            return GeminiConnector(api_key, model, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")