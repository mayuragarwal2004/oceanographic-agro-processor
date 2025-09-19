"""
GROQ API Connector Implementation
Fast inference with open-source models like Llama, Mixtral, etc.
"""

import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any
from .base import BaseLLMConnector, LLMMessage, LLMResponse

class GroqConnector(BaseLLMConnector):
    """GROQ API connector implementation"""
    
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Default timeout settings
        self.timeout = kwargs.get('timeout', 30)
        self.max_retries = kwargs.get('max_retries', 3)
    
    async def generate(
        self, 
        messages: List[LLMMessage], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using GROQ API"""
        
        # Convert messages to GROQ format
        groq_messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": groq_messages,
            "temperature": temperature,
            "stream": False
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        # Add any additional parameters
        payload.update(kwargs)
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_response(data)
                        elif response.status == 429:  # Rate limit
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                raise Exception(f"Rate limited after {self.max_retries} attempts")
                        else:
                            error_text = await response.text()
                            raise Exception(f"GROQ API error {response.status}: {error_text}")
            
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    raise Exception(f"Timeout after {self.max_retries} attempts")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    raise e
    
    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Parse GROQ API response"""
        choice = data["choices"][0]
        content = choice["message"]["content"]
        
        # Extract usage information if available
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens")
        
        return LLMResponse(
            content=content,
            tokens_used=tokens_used,
            model_used=data.get("model"),
            metadata={
                "finish_reason": choice.get("finish_reason"),
                "usage": usage
            }
        )
    
    def validate_connection(self) -> bool:
        """Validate GROQ API connection"""
        try:
            # Simple sync validation
            import requests
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_available_models(self) -> List[str]:
        """List available models from GROQ"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["id"] for model in data["data"]]
                    return []
        except Exception:
            return []

# Commonly used GROQ models
GROQ_MODELS = {
    "mixtral-8x7b": "mixtral-8x7b-32768",
    "llama-70b": "llama2-70b-4096", 
    "llama-7b": "llama2-7b-2048",
    "gemma-7b": "gemma-7b-it",
    "code-llama": "codellama-34b-instruct"
}