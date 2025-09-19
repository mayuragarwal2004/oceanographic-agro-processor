"""
Query Understanding Agent
Extracts entities (location, time, parameter, comparison intent) from user queries
Maps natural language to an operator graph instead of direct SQL
"""

import json
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .base_agent import BaseAgent, AgentResult, LLMMessage

class ComparisonIntent(Enum):
    """Types of comparison intents"""
    NONE = "none"
    TEMPORAL = "temporal"  # Compare across time
    SPATIAL = "spatial"   # Compare across locations
    PARAMETER = "parameter"  # Compare different parameters
    STATISTICAL = "statistical"  # Statistical analysis (avg, max, min, etc.)

class TemporalIntent(Enum):
    """Types of temporal queries"""
    POINT = "point"          # Specific date/time
    RANGE = "range"          # Date range
    RECENT = "recent"        # Recent data (last N days/months)
    SEASONAL = "seasonal"    # Seasonal analysis
    TREND = "trend"          # Trend analysis
    CLIMATOLOGY = "climatology"  # Long-term averages

@dataclass
class OperatorNode:
    """Node in the operator graph representing a query operation"""
    operation: str
    parameters: Dict[str, Any]
    dependencies: List[str]  # IDs of nodes this depends on
    node_id: str

@dataclass 
class ExtractedEntities:
    """Entities extracted from the user query"""
    locations: List[str]
    parameters: List[str]  # temp, salinity, pressure
    time_range: Optional[Dict[str, Any]]
    depth_range: Optional[Dict[str, float]]
    comparison_intent: ComparisonIntent
    temporal_intent: TemporalIntent
    statistical_operations: List[str]
    confidence_score: float

@dataclass
class OperatorGraph:
    """Graph of operations to execute for the query"""
    nodes: List[OperatorNode]
    root_node_id: str
    metadata: Dict[str, Any]

class QueryUnderstandingAgent(BaseAgent):
    """Agent responsible for understanding and parsing user queries"""
    
    def __init__(self, config):
        super().__init__(config, "query_understanding")
        
        # Known oceanographic parameters
        self.known_parameters = {
            'temperature': ['temp', 'temperature', 'sst', 'sea surface temperature'],
            'salinity': ['salinity', 'psal', 'salt', 'practical salinity'],
            'pressure': ['pressure', 'pres', 'depth', 'sea pressure']
        }
        
        # Statistical operations
        self.statistical_ops = {
            'average': ['average', 'avg', 'mean'],
            'maximum': ['max', 'maximum', 'highest', 'peak'],
            'minimum': ['min', 'minimum', 'lowest'],
            'standard_deviation': ['std', 'stdev', 'standard deviation', 'variation'],
            'median': ['median', 'middle'],
            'trend': ['trend', 'trending', 'change', 'evolution'],
        }
    
    def sanitize_json(self, text: str) -> str:
        """
        Sanitize JSON response by extracting only the JSON part
        Removes markdown formatting, code blocks, and extra text
        """
        if not text or not isinstance(text, str):
            return "{}"
        
        text = text.strip()
        
        # Remove markdown code block formatting
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
        
        # Find JSON object boundaries
        # Look for the first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            # Extract the JSON part
            json_part = text[first_brace:last_brace + 1]
            
            # Clean up common issues
            json_part = json_part.replace('\n', ' ')  # Replace newlines with spaces
            json_part = re.sub(r'\s+', ' ', json_part)  # Normalize whitespace
            json_part = json_part.replace("'", '"')  # Replace single quotes with double quotes
            
            # Remove trailing commas before closing braces/brackets
            json_part = re.sub(r',\s*}', '}', json_part)
            json_part = re.sub(r',\s*]', ']', json_part)
            
            return json_part.strip()
        
        # If no braces found, try to find array notation
        first_bracket = text.find('[')
        last_bracket = text.rfind(']')
        
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            json_part = text[first_bracket:last_bracket + 1]
            json_part = json_part.replace('\n', ' ')
            json_part = re.sub(r'\s+', ' ', json_part)
            json_part = json_part.replace("'", '"')
            json_part = re.sub(r',\s*]', ']', json_part)
            return json_part.strip()
        
        # If nothing found, return empty object
        return "{}"
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Process a user query and extract structured information"""
        
        try:
            if not isinstance(input_data, str):
                return AgentResult.error_result(
                    self.agent_name,
                    ["Input must be a string query"]
                )
            
            user_query = input_data.strip()
            self.logger.info(f"Processing query: {user_query[:100]}...")
            
            # Extract entities using LLM
            entities = await self._extract_entities_with_llm(user_query)
            
            # Generate operator graph
            operator_graph = await self._generate_operator_graph(entities, user_query)
            
            return AgentResult.success_result(
                self.agent_name,
                {
                    'entities': entities,
                    'operator_graph': operator_graph,
                    'original_query': user_query
                },
                {'confidence': entities.confidence_score}
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to process query: {str(e)}"]
            )
    
    async def _extract_entities_with_llm(self, query: str) -> ExtractedEntities:
        """Use LLM to extract entities from the query"""
        
        system_prompt = """You are an expert at understanding oceanographic data queries. 
Extract structured information from user queries about ocean temperature, salinity, and pressure data.

Return a JSON object with the following structure:
{
    "locations": ["location1", "location2"],  // Geographic locations mentioned
    "parameters": ["temperature", "salinity", "pressure"],  // Which ocean parameters
    "time_range": {
        "type": "point|range|recent|seasonal|climatology",
        "start": "YYYY-MM-DD",  // if applicable
        "end": "YYYY-MM-DD",    // if applicable
        "description": "textual description of time intent"
    },
    "depth_range": {
        "min": 0,     // meters (if specified)
        "max": 1000,  // meters (if specified)
        "surface_only": true/false
    },
    "comparison_intent": "none|temporal|spatial|parameter|statistical",
    "temporal_intent": "point|range|recent|seasonal|trend|climatology",
    "statistical_operations": ["average", "maximum", "trend"],
    "confidence_score": 0.95  // How confident you are (0-1)
}

Examples:
- "Temperature in the Pacific Ocean last month" -> locations: ["Pacific Ocean"], parameters: ["temperature"], temporal_intent: "recent"
- "Compare salinity between Atlantic and Pacific" -> locations: ["Atlantic", "Pacific"], parameters: ["salinity"], comparison_intent: "spatial"
- "Average temperature trends in Indian Ocean from 2020 to 2023" -> locations: ["Indian Ocean"], parameters: ["temperature"], comparison_intent: "statistical", statistical_operations: ["average", "trend"]"""
        
        user_prompt = f"Extract entities from this oceanographic query: \"{query}\""
        
        messages = [
            self.create_system_message(system_prompt),
            self.create_user_message(user_prompt)
        ]
        
        response = await self.call_llm(messages, temperature=0.1)  # Low temperature for consistency
        
        try:
            # Parse JSON response using sanitized content
            sanitized_content = self.sanitize_json(response.content)
            entities_data = json.loads(sanitized_content)
            
            return ExtractedEntities(
                locations=entities_data.get('locations', []),
                parameters=entities_data.get('parameters', []),
                time_range=entities_data.get('time_range'),
                depth_range=entities_data.get('depth_range'),
                comparison_intent=ComparisonIntent(entities_data.get('comparison_intent', 'none')),
                temporal_intent=TemporalIntent(entities_data.get('temporal_intent', 'point')),
                statistical_operations=entities_data.get('statistical_operations', []),
                confidence_score=entities_data.get('confidence_score', 0.7)
            )
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM JSON response: {e}")
            # Fallback to rule-based extraction
            return self._extract_entities_rule_based(query)
    
    def _extract_entities_rule_based(self, query: str) -> ExtractedEntities:
        """Fallback rule-based entity extraction"""
        
        query_lower = query.lower()
        
        # Extract locations (basic patterns)
        locations = []
        location_patterns = [
            r'\b(pacific|atlantic|indian|southern|arctic)\s+ocean\b',
            r'\b(mediterranean|caribbean|baltic|red)\s+sea\b',
            r'\b(gulf\s+of\s+\w+)\b',
            r'\b(bay\s+of\s+\w+)\b'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, query_lower)
            locations.extend(matches)
        
        # Extract parameters
        parameters = []
        for param, aliases in self.known_parameters.items():
            if any(alias in query_lower for alias in aliases):
                parameters.append(param)
        
        # Extract statistical operations
        stats_ops = []
        for op, aliases in self.statistical_ops.items():
            if any(alias in query_lower for alias in aliases):
                stats_ops.append(op)
        
        # Determine comparison intent
        comparison_intent = ComparisonIntent.NONE
        if any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs', 'between']):
            if len(locations) > 1:
                comparison_intent = ComparisonIntent.SPATIAL
            elif len(parameters) > 1:
                comparison_intent = ComparisonIntent.PARAMETER
            else:
                comparison_intent = ComparisonIntent.TEMPORAL
        elif stats_ops:
            comparison_intent = ComparisonIntent.STATISTICAL
        
        # Determine temporal intent
        temporal_intent = TemporalIntent.POINT
        if any(word in query_lower for word in ['trend', 'change', 'evolution']):
            temporal_intent = TemporalIntent.TREND
        elif any(word in query_lower for word in ['recent', 'last', 'past']):
            temporal_intent = TemporalIntent.RECENT
        elif any(word in query_lower for word in ['range', 'from', 'to', 'between']):
            temporal_intent = TemporalIntent.RANGE
        
        return ExtractedEntities(
            locations=locations,
            parameters=parameters or ['temperature'],  # Default to temperature
            time_range=None,
            depth_range=None,
            comparison_intent=comparison_intent,
            temporal_intent=temporal_intent,
            statistical_operations=stats_ops,
            confidence_score=0.6  # Lower confidence for rule-based
        )
    
    async def _generate_operator_graph(self, entities: ExtractedEntities, query: str) -> OperatorGraph:
        """Generate an operator graph from extracted entities"""
        
        nodes = []
        node_counter = 0
        
        def create_node(operation: str, parameters: Dict[str, Any], dependencies: List[str] = None) -> OperatorNode:
            nonlocal node_counter
            node_id = f"op_{node_counter}"
            node_counter += 1
            return OperatorNode(
                operation=operation,
                parameters=parameters,
                dependencies=dependencies or [],
                node_id=node_id
            )
        
        # 1. Data retrieval node (always needed)
        retrieval_node = create_node(
            "data_retrieval",
            {
                "locations": entities.locations,
                "parameters": entities.parameters,
                "time_range": entities.time_range,
                "depth_range": entities.depth_range
            }
        )
        nodes.append(retrieval_node)
        current_node_id = retrieval_node.node_id
        
        # 2. Statistical operations (if needed)
        if entities.statistical_operations:
            stats_node = create_node(
                "statistical_analysis",
                {
                    "operations": entities.statistical_operations,
                    "parameters": entities.parameters
                },
                [current_node_id]
            )
            nodes.append(stats_node)
            current_node_id = stats_node.node_id
        
        # 3. Comparison operations (if needed)
        if entities.comparison_intent != ComparisonIntent.NONE:
            comparison_node = create_node(
                "comparison",
                {
                    "type": entities.comparison_intent.value,
                    "locations": entities.locations,
                    "parameters": entities.parameters
                },
                [current_node_id]
            )
            nodes.append(comparison_node)
            current_node_id = comparison_node.node_id
        
        # 4. Visualization node (always add as final step)
        viz_node = create_node(
            "visualization",
            {
                "type": self._determine_visualization_type(entities),
                "parameters": entities.parameters,
                "locations": entities.locations
            },
            [current_node_id]
        )
        nodes.append(viz_node)
        
        return OperatorGraph(
            nodes=nodes,
            root_node_id=viz_node.node_id,
            metadata={
                "query": query,
                "confidence": entities.confidence_score,
                "complexity_score": len(nodes)
            }
        )
    
    def _determine_visualization_type(self, entities: ExtractedEntities) -> str:
        """Determine the best visualization type for the query"""
        
        if entities.comparison_intent == ComparisonIntent.SPATIAL and len(entities.locations) > 1:
            return "map_comparison"
        elif entities.temporal_intent == TemporalIntent.TREND:
            return "time_series"
        elif entities.comparison_intent == ComparisonIntent.PARAMETER and len(entities.parameters) > 1:
            return "parameter_comparison"
        elif len(entities.locations) == 1 and len(entities.parameters) == 1:
            return "depth_profile"
        else:
            return "summary_map"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about query understanding capabilities"""
        return {
            "description": "Understands natural language queries about oceanographic data",
            "supported_parameters": list(self.known_parameters.keys()),
            "supported_operations": list(self.statistical_ops.keys()),
            "comparison_types": [intent.value for intent in ComparisonIntent],
            "temporal_types": [intent.value for intent in TemporalIntent],
            "confidence_threshold": 0.5
        }