"""
Agents Package
Multi-agent system for oceanographic data analysis
"""

from .base_agent import BaseAgent, AgentResult, LLMMessage
from .config import AgentSystemConfig
from .orchestrator import AgentOrchestrator

# Individual agents
from .query_understanding import QueryUnderstandingAgent
from .geospatial import GeospatialAgent
from .data_retrieval import DataRetrievalAgent
from .analysis import AnalysisAgent
from .visualization import VisualizationAgent
from .critic import CriticAgent
from .conversation import ConversationAgent

# LLM connectors
from .llm.base import BaseLLMConnector, LLMConnectorFactory
from .llm.groq_connector import GroqConnector

__version__ = "1.0.0"

__all__ = [
    # Core classes
    'BaseAgent',
    'AgentResult', 
    'LLMMessage',
    'AgentSystemConfig',
    'AgentOrchestrator',
    
    # Individual agents
    'QueryUnderstandingAgent',
    'GeospatialAgent', 
    'DataRetrievalAgent',
    'AnalysisAgent',
    'VisualizationAgent',
    'CriticAgent',
    'ConversationAgent',
    
    # LLM connectors
    'BaseLLMConnector',
    'LLMConnectorFactory',
    'GroqConnector'
]