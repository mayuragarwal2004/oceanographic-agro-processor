"""
Base Agent Class
Provides common functionality for all agents in the system
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .llm import BaseLLMConnector, LLMMessage, LLMResponse, LLMConnectorFactory
from .config import AgentSystemConfig


def extract_json_string(s: str) -> Optional[str]:
    """
    Extracts the JSON substring from a string by finding the first '{'
    and the last '}'.
    Returns None if no valid JSON block is found.
    """
    start = s.find("{")
    end = s.rfind("}")
    
    if start == -1 or end == -1 or start > end:
        return None  # No valid JSON found
    
    return s[start:end+1]


@dataclass
class AgentResult:
    """Result returned by an agent"""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    errors: List[str]
    agent_name: str
    
    @classmethod
    def success_result(
        cls, agent_name: str, data: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> 'AgentResult':
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            errors=[],
            agent_name=agent_name
        )
    
    @classmethod
    def error_result(
        cls, agent_name: str, errors: List[str], data: Any = None
    ) -> 'AgentResult':
        return cls(
            success=False,
            data=data,
            metadata={},
            errors=errors,
            agent_name=agent_name
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization/logging"""
        return {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
            "errors": self.errors,
            "agent_name": self.agent_name,
        }


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, config: AgentSystemConfig, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agent.{agent_name}")
        
        # Initialize LLM connector
        self.llm: BaseLLMConnector = LLMConnectorFactory.create_connector(
            provider=config.llm_config.provider,
            api_key=config.llm_config.api_key,
            model=config.llm_config.model,
            timeout=config.llm_config.timeout,
            max_retries=config.llm_config.max_retries
        )
        
        self._setup_logging()
    
    @classmethod
    def from_config(cls, config: AgentSystemConfig) -> 'BaseAgent':
        """
        Factory method for creating an agent from configuration.
        Subclasses can override this if they need custom init logic.
        """
        return cls(config=config, agent_name=cls.__name__)
    
    def _setup_logging(self):
        """Set up logging for the agent"""
        if self.config.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    async def call_llm(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Call the LLM with error handling and logging"""
        
        temp = temperature or self.config.llm_config.temperature
        tokens = max_tokens or self.config.llm_config.max_tokens
        
        # Log the raw input to LLM
        self.logger.info(f"=== LLM INPUT for {self.agent_name} ===")
        for i, message in enumerate(messages):
            self.logger.info(f"Message {i+1} [{message.role}]:")
            self.logger.info(f"{message.content}")
            self.logger.info(f"--- End Message {i+1} ---")
        self.logger.info(f"Temperature: {temp}, Max Tokens: {tokens}")
        self.logger.info(f"=== END LLM INPUT ===")
        
        try:
            response = await self.llm.generate(
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                **kwargs
            )
            
            # Log the raw output from LLM
            self.logger.info(f"=== LLM RAW OUTPUT for {self.agent_name} ===")
            self.logger.info(f"Content ({len(response.content)} chars):")
            self.logger.info(f"{response.content}")
            if hasattr(response, 'usage') and response.usage:
                self.logger.info(f"Usage: {response.usage}")
            self.logger.info(f"=== END LLM RAW OUTPUT ===")
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise e

    def create_system_message(self, content: str) -> LLMMessage:
        """Create a system message"""
        return LLMMessage(role="system", content=content)
    
    def create_user_message(self, content: str) -> LLMMessage:
        """Create a user message"""
        return LLMMessage(role="user", content=content)
    
    def create_assistant_message(self, content: str) -> LLMMessage:
        """Create an assistant message"""
        return LLMMessage(role="assistant", content=content)
    
    @abstractmethod
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process input data and return a result
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return information about what this agent can do
        Must be implemented by subclasses.
        """
        pass
    
    def validate_input(self, input_data: Any) -> List[str]:
        """
        Validate input data and return list of errors
        Subclasses can override this with stricter validation.
        """
        return []  # Base implementation accepts all input
    
    async def health_check(self) -> bool:
        """
        Check if the agent is healthy and ready to process requests
        """
        try:
            # Test LLM connection
            if not self.llm.validate_connection():
                return False
            
            # Agent-specific health checks can be overridden
            return await self._agent_health_check()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def _agent_health_check(self) -> bool:
        """Override this for agent-specific health checks"""
        return True

    async def shutdown(self):
        """
        Clean up resources when the agent is shutting down.
        Subclasses can extend this to close DB connections, stop tasks, etc.
        """
        if hasattr(self.llm, "close"):
            await self.llm.close()
        self.logger.info(f"Agent {self.agent_name} shut down cleanly.")
