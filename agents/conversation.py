"""
Conversation Agent
Handles natural language responses and manages conversation flow with users
Synthesizes results from all other agents into coherent, user-friendly responses
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .base_agent import BaseAgent, AgentResult, LLMMessage

class ResponseType(Enum):
    """Types of responses to generate"""
    SUMMARY = "summary"                   # Executive summary
    DETAILED = "detailed"                 # Comprehensive analysis
    TECHNICAL = "technical"               # Technical details for experts
    EDUCATIONAL = "educational"           # Educational/explanatory
    EXPLORATORY = "exploratory"           # Suggesting further analysis

class ConversationContext(Enum):
    """Context for conversation management"""
    INITIAL_QUERY = "initial_query"      # First query in session
    FOLLOW_UP = "follow_up"              # Follow-up questions
    CLARIFICATION = "clarification"       # User asking for clarification
    DEEP_DIVE = "deep_dive"              # User wants more detail
    NEW_TOPIC = "new_topic"              # Switching to new topic

@dataclass
class ConversationState:
    """State management for ongoing conversation"""
    session_id: str
    previous_queries: List[str]
    previous_results: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    context: ConversationContext
    current_topic: Optional[str] = None

@dataclass
class ResponseComponents:
    """Components of a conversation response"""
    executive_summary: str
    key_findings: List[str]
    technical_details: Optional[str]
    visualizations: List[Dict[str, Any]]
    recommendations: List[str]
    caveats: List[str]
    follow_up_suggestions: List[str]

class ConversationAgent(BaseAgent):
    """Agent responsible for natural language conversation with users"""
    
    def __init__(self, config):
        super().__init__(config, "conversation")
        
        # Response templates
        self.response_templates = {
            'summary_intro': "Based on your query about {topic}, I've analyzed {data_count} oceanographic records.",
            'findings_intro': "Here are the key findings:",
            'technical_intro': "Technical details:",
            'recommendations_intro': "Recommendations:",
            'limitations_intro': "Important limitations to consider:",
            'follow_up_intro': "You might also be interested in:"
        }
        
        # Common oceanographic explanations
        self.explanations = {
            'temperature': "Ocean temperature affects water density, marine ecosystems, and climate patterns.",
            'salinity': "Salinity measures dissolved salt content and influences water density and circulation.",
            'pressure': "Pressure increases with depth and is measured in decibars (dbar), roughly equivalent to meters.",
            'quality_flags': "Quality control flags indicate data reliability: 1=good, 2=probably good, 3=probably bad, 4=bad.",
            'argo_floats': "Argo floats are autonomous instruments that measure temperature, salinity, and pressure profiles."
        }
        
        # Response style preferences
        self.style_guidelines = {
            'technical': {
                'include_statistics': True,
                'include_uncertainties': True,
                'use_scientific_terminology': True,
                'detail_level': 'high'
            },
            'general': {
                'include_statistics': False,
                'include_uncertainties': False,
                'use_scientific_terminology': False,
                'detail_level': 'medium'
            },
            'educational': {
                'include_explanations': True,
                'include_context': True,
                'use_analogies': True,
                'detail_level': 'medium'
            }
        }
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Generate natural language response from analysis results"""
        
        print("test1")
        
        try:
            if not isinstance(input_data, dict):
                return AgentResult.error_result(
                    self.agent_name,
                    ["Input must be a dictionary containing analysis results"]
                )
            print("test2")
            
            # Extract all component results
            query_results = input_data.get('query_understanding', {})
            geospatial_results = input_data.get('geospatial', {})
            data_results = input_data.get('data_retrieval', {})
            analysis_results = input_data.get('analysis', {})
            visualization_results = input_data.get('visualization', {})
            validation_results = input_data.get('validation', {})
            
            print("test3")
            
            # Get conversation context
            conversation_context = self._determine_conversation_context(context or {})
            user_preferences = context.get('user_preferences', {})
            
            print("test4")
            
            self.logger.info(f"Generating {conversation_context.value} response")
            
            print("test5")
            
            # Synthesize response components
            response_components = await self._synthesize_response_components(
                query_results, geospatial_results, data_results,
                analysis_results, visualization_results, validation_results
            )
            
            print("test6")
            
            # Generate natural language response
            natural_response = await self._generate_natural_response(
                response_components, conversation_context, user_preferences
            )
            
            print("test7")
            
            # Create follow-up suggestions
            follow_ups = await self._generate_follow_up_suggestions(
                response_components, query_results
            )
            
            print("test8")
            
            return AgentResult.success_result(
                self.agent_name,
                {
                    'response': natural_response,
                    'components': {
                        'executive_summary': response_components.executive_summary,
                        'key_findings': response_components.key_findings,
                        'recommendations': response_components.recommendations,
                        'caveats': response_components.caveats
                    },
                    'visualizations': response_components.visualizations,
                    'follow_up_suggestions': follow_ups,
                    'metadata': {
                        'conversation_context': conversation_context.value,
                        'response_length': len(natural_response),
                        'components_count': len(response_components.key_findings)
                    }
                },
                {'response_generated': True}
            )
            
        except Exception as e:
            self.logger.error(f"Error generating conversation response: {str(e)}")
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to generate response: {str(e)}"]
            )
    
    def _determine_conversation_context(self, context: Dict[str, Any]) -> ConversationContext:
        """Determine the conversation context"""
        
        # Check for explicit context
        if 'conversation_context' in context:
            try:
                return ConversationContext(context['conversation_context'])
            except ValueError:
                pass
        
        # Infer from context
        if context.get('is_follow_up', False):
            return ConversationContext.FOLLOW_UP
        elif context.get('needs_clarification', False):
            return ConversationContext.CLARIFICATION
        elif context.get('deep_dive_requested', False):
            return ConversationContext.DEEP_DIVE
        else:
            return ConversationContext.INITIAL_QUERY
    
    async def _synthesize_response_components(self, query_results: Dict[str, Any],
                                            geospatial_results: Dict[str, Any],
                                            data_results: Dict[str, Any],
                                            analysis_results: Dict[str, Any],
                                            visualization_results: Dict[str, Any],
                                            validation_results: Dict[str, Any]) -> ResponseComponents:
        """Synthesize all agent results into response components"""
        
        print("test1.1")
        
        # Extract key information
        original_query = query_results.get('original_query', 'your query')
        locations = geospatial_results.get('locations', [])
        data_count = data_results.get('metadata', {}).get('row_count', 0)
        
        print("test1.2")
        
        # Generate executive summary
        exec_summary = await self._create_executive_summary(
            original_query, locations, data_count, analysis_results
        )
        
        print("test1.3")
        
        # Extract key findings
        key_findings = self._extract_key_findings(analysis_results, validation_results)
        
        print("test1.4")
        
        # Format visualizations
        visualizations = self._format_visualizations(visualization_results)
        self.logger.info(f"Formatted {len(visualizations)} visualizations for response")
        self.logger.info(f"Visualizations: {visualizations}")
        print("test1.5")
        
        # Generate recommendations
        self.logger.info(f"analysis_results keys: {list(analysis_results.keys())}")
        self.logger.info(f"validation_results keys: {list(validation_results.keys())}")
        self.logger.info(f"query_results keys: {list(query_results.keys())}")
        recommendations = self._compile_recommendations(
            analysis_results, validation_results, query_results
        )
        
        print("test1.6")
        
        # Identify caveats and limitations
        caveats = self._identify_caveats(validation_results, data_results)
        
        print("test1.7")
        
        # Generate follow-up suggestions
        self.logger.info(f"query_results: {query_results}")
        self.logger.info(f"analysis_results: {analysis_results}")
        follow_ups = self._suggest_follow_ups(query_results, analysis_results)
        self.logger.info(f"Follow-up suggestions: {follow_ups}")    
        print("test1.8")
        
        return ResponseComponents(
            executive_summary=exec_summary,
            key_findings=key_findings,
            technical_details=None,  # Generated separately if needed
            visualizations=visualizations,
            recommendations=recommendations,
            caveats=caveats,
            follow_up_suggestions=follow_ups
        )
    
    async def _create_executive_summary(self, query: str, locations: List[Dict[str, Any]], 
                                      data_count: int, analysis_results: Dict[str, Any]) -> str:
        """Create executive summary using LLM"""
        
        print("test2.1")
        
        print("analysis_results")
        print(analysis_results)
        
        # Prepare summary data
        summary_context = {
            'original_query': query,
            'locations_analyzed': [loc.get('name', 'Unknown') for loc in locations],
            'data_points': data_count,
            'analysis_types': list(analysis_results.keys()) #error here
        }
        
        print("test2.2")
        
        # Extract key statistics from nested analysis_results structure
        actual_analysis = analysis_results.get('analysis_results', {})
        if 'descriptive' in actual_analysis:
            desc_stats = actual_analysis['descriptive']
            summary_context['parameters'] = list(desc_stats.keys())
            
            # Add notable statistics
            notable_stats = {}
            for param, stats in desc_stats.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    notable_stats[param] = {
                        'average': round(stats['mean'], 2),
                        'range': f"{round(stats.get('min', 0), 2)} - {round(stats.get('max', 0), 2)}"
                    }
            summary_context['statistics'] = notable_stats
        
        system_prompt = """You are an oceanographic data analyst. Create a clear, concise executive summary 
        of the analysis results. The summary should be 2-3 sentences and accessible to non-experts while 
        being scientifically accurate.
        
        Include:
        - What was analyzed (location, parameters, data scope)
        - Key findings or patterns with ACTUAL VALUES from the statistics (never use placeholders like [insert value])
        - Context for significance
        
        IMPORTANT: If you have statistics in the data, use the actual numerical values. Never use placeholders.
        
        Keep it conversational and informative."""
        
        user_prompt = f"Create an executive summary for this oceanographic analysis: {json.dumps(summary_context, default=str)}"
        
        try:
            messages = [
                self.create_system_message(system_prompt),
                self.create_user_message(user_prompt)
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            return response.content.strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate executive summary with LLM: {e}")
            # Fallback to template
            locations_str = ", ".join([loc.get('name', 'Unknown') for loc in locations])
            return f"I analyzed {data_count} oceanographic records from {locations_str}. " + \
                   f"The analysis included {', '.join(analysis_results.keys())} of ocean parameters."
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any], 
                             validation_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis results"""
        
        findings = []
        
        # Access nested analysis_results structure
        actual_analysis = analysis_results.get('analysis_results', {})
        
        # Descriptive statistics findings
        if 'descriptive' in actual_analysis:
            desc_stats = actual_analysis['descriptive']
            for param, stats in desc_stats.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    param_name = self._get_parameter_name(param)
                    findings.append(f"{param_name} averaged {stats['mean']:.2f}°C across all measurements")
        
        # Trend findings
        if 'trend' in actual_analysis:
            trend_results = actual_analysis['trend']
            for param, trend in trend_results.items():
                if isinstance(trend, dict) and 'trend_direction' in trend:
                    param_name = self._get_parameter_name(param)
                    direction = trend['trend_direction']
                    if trend.get('significance') == 'significant':
                        findings.append(f"{param_name} shows a statistically significant {direction} trend over time")
        
        # Anomaly findings
        if 'anomaly' in actual_analysis:
            anomaly_results = actual_analysis['anomaly']
            for param, anomalies in anomaly_results.items():
                if isinstance(anomalies, dict) and 'anomaly_percentage' in anomalies:
                    if anomalies['anomaly_percentage'] > 5:  # >5% anomalies worth mentioning
                        param_name = self._get_parameter_name(param)
                        findings.append(f"{anomalies['anomaly_percentage']:.1f}% of {param_name} measurements were anomalous")
        
        # Validation findings
        if validation_results and 'validation_report' in validation_results:
            report = validation_results['validation_report']
            if report.get('overall_score', 100) < 90:
                findings.append(f"Data quality score: {report['overall_score']:.1f}% - some quality issues identified")
        
        return findings[:5]  # Limit to top 5 findings
    
    def _format_visualizations(self, visualization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format visualization information for response"""
        
        if not visualization_results or 'visualizations' not in visualization_results:
            return []
        
        vizs = visualization_results['visualizations']
        formatted = []
        
        for viz_type, viz_data in vizs.items():
            formatted.append({
                'type': viz_type,
                'title': viz_data.get('metadata', {}).get('title', viz_type.replace('_', ' ').title()),
                'description': self._get_viz_description(viz_type),
                'available': True
            })
        
        return formatted
    
    def _get_viz_description(self, viz_type: str) -> str:
        """Get description for visualization type"""
        
        descriptions = {
            'map': 'Interactive map showing data locations and values',
            'time_series': 'Time series plot showing changes over time',
            'depth_profile': 'Vertical profile showing variation with depth',
            'histogram': 'Distribution histogram of parameter values',
            'correlation_matrix': 'Correlation heatmap between parameters',
            'comparison': 'Box plot comparing parameter distributions',
            'dashboard': 'Combined dashboard with multiple visualizations'
        }
        
        return descriptions.get(viz_type, f'{viz_type.replace("_", " ").title()} visualization')
    
    def _compile_recommendations(self, analysis_results: Dict[str, Any],
                               validation_results: Dict[str, Any],
                               query_results: Dict[str, Any]) -> List[str]:
        """Compile recommendations from all agents"""
        
        recommendations = []
        
        # Analysis insights as recommendations
        if 'insights' in analysis_results:
            analysis_insights = analysis_results['insights']
            if isinstance(analysis_insights, list):
                recommendations.extend(analysis_insights[:3])  # Top 3 insights
        
        # Validation recommendations
        if validation_results and 'validation_report' in validation_results:
            val_recs = validation_results['validation_report'].get('recommendations', [])
            recommendations.extend(val_recs[:2])  # Top 2 validation recommendations
        
        # Query-based recommendations
        # Query-based recommendations
        entities = query_results.get('entities')
        # Check if entities exists and has the attributes we need
        if entities and hasattr(entities, 'temporal_intent'):
            # FIX: Use dot notation to access attributes of the ExtractedEntities object
            if entities.temporal_intent == 'recent':
                recommendations.append("Consider analyzing longer time periods to identify seasonal patterns")
            elif entities.comparison_intent == 'spatial':
                recommendations.append("Explore temporal trends in each region for deeper insights")
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _identify_caveats(self, validation_results: Dict[str, Any], 
                         data_results: Dict[str, Any]) -> List[str]:
        """Identify important caveats and limitations"""
        
        caveats = []
        
        # Data size limitations
        data_count = data_results.get('metadata', {}).get('row_count', 0)
        if data_count < 100:
            caveats.append(f"Analysis based on limited data ({data_count} records) - results should be interpreted cautiously")
        
        # Validation issues
        if validation_results and 'validation_report' in validation_results:
            report = validation_results['validation_report']
            
            if not report.get('approved', True):
                caveats.append("Data quality validation identified issues that may affect reliability")
            
            critical_issues = report.get('summary', {}).get('issue_counts', {}).get('critical', 0)
            if critical_issues > 0:
                caveats.append(f"{critical_issues} critical data quality issues were identified")
        
        # Temporal limitations
        if 'metadata' in data_results:
            metadata = data_results['metadata']
            if 'time_range' in metadata:
                caveats.append("Results represent conditions only for the analyzed time period")
        
        return caveats[:3]  # Limit to 3 caveats
    
    def _suggest_follow_ups(self, query_results: Dict[str, Any], 
                           analysis_results: Dict[str, Any]) -> List[str]:
        """Suggest follow-up analyses or questions"""
        
        suggestions = []
        
        # FIX: Access the nested 'analysis_results' dictionary correctly
        actual_analysis = analysis_results.get('analysis_results', {})
        
        # Based on current analysis
        if 'descriptive' in actual_analysis and 'trend' not in actual_analysis:
            suggestions.append("Analyze trends over time to understand temporal patterns")
        
        if 'trend' in actual_analysis and 'seasonal' not in actual_analysis:
            suggestions.append("Explore seasonal patterns in the data")
        
        # Based on query entities
        entities = query_results.get('entities')
        
        # FIX: Use dot notation (.) to access attributes of the ExtractedEntities object
        if entities and hasattr(entities, 'locations'):
            locations = entities.locations or []
            parameters = entities.parameters or []
            
            if len(locations) == 1 and len(parameters) >= 2:
                suggestions.append("Compare these parameters across different ocean regions")
            
            if len(parameters) == 1 and len(locations) >= 2:
                suggestions.append("Analyze this parameter's relationship with depth or season")
        
        suggestions.extend([
            "Examine correlations between different oceanographic parameters",
            "Investigate anomalous values for potential climate signals",
            "Compare results with climatological averages"
        ])
        
        return suggestions[:4]
        
    async def _generate_natural_response(self, components: ResponseComponents,
                                       context: ConversationContext,
                                       preferences: Dict[str, Any]) -> str:
        """Generate the final natural language response"""
        
        # Determine response style
        style = preferences.get('style', 'general')
        detail_level = preferences.get('detail_level', 'medium')
        
        # Build response sections
        response_parts = []
        
        # Executive summary
        response_parts.append(components.executive_summary)
        
        # Key findings
        if components.key_findings:
            response_parts.append(f"\n{self.response_templates['findings_intro']}")
            for i, finding in enumerate(components.key_findings, 1):
                response_parts.append(f"{i}. {finding}")
        
        # Visualizations available
        if components.visualizations:
            viz_list = ", ".join([viz['title'] for viz in components.visualizations])
            response_parts.append(f"\nI've created visualizations including: {viz_list}")
        
        # Recommendations
        if components.recommendations:
            response_parts.append(f"\n{self.response_templates['recommendations_intro']}")
            for rec in components.recommendations:
                response_parts.append(f"• {rec}")
        
        # Caveats
        if components.caveats:
            response_parts.append(f"\n{self.response_templates['limitations_intro']}")
            for caveat in components.caveats:
                response_parts.append(f"• {caveat}")
        
        # Follow-up suggestions
        if components.follow_up_suggestions:
            response_parts.append(f"\n{self.response_templates['follow_up_intro']}")
            for suggestion in components.follow_up_suggestions[:3]:
                response_parts.append(f"• {suggestion}")
        
        return "\n".join(response_parts)
    
    async def _generate_follow_up_suggestions(self, components: ResponseComponents,
                                            query_results: Dict[str, Any]) -> List[str]:
        """Generate intelligent follow-up suggestions"""
        
        # Use LLM to generate contextual follow-ups
        context_data = {
            'findings_count': len(components.key_findings),
            'has_trends': any('trend' in finding.lower() for finding in components.key_findings),
            'has_anomalies': any('anomal' in finding.lower() for finding in components.key_findings),
            'original_query': query_results.get('original_query', '')
        }
        
        system_prompt = """Based on the oceanographic analysis results, suggest 3 intelligent follow-up 
        questions or analyses that would provide deeper insights. Make suggestions specific and actionable.
        
        Focus on:
        - Deeper scientific understanding
        - Related phenomena to investigate
        - Different analytical approaches
        - Broader temporal or spatial context
        """
        
        user_prompt = f"Suggest follow-ups for this analysis context: {json.dumps(context_data)}"
        
        try:
            messages = [
                self.create_system_message(system_prompt),
                self.create_user_message(user_prompt)
            ]
            
            response = await self.call_llm(messages, temperature=0.4)
            
            # Parse suggestions
            suggestions = [line.strip().lstrip('•-1234567890. ') 
                          for line in response.content.split('\n') 
                          if line.strip() and not line.strip().isdigit()]
            
            return suggestions[:3]
            
        except Exception as e:
            self.logger.warning(f"Failed to generate follow-up suggestions: {e}")
            return components.follow_up_suggestions[:3]
    
    def _get_parameter_name(self, param: str) -> str:
        """Get human-readable parameter name"""
        
        name_mapping = {
            'TEMP': 'Temperature',
            'PSAL': 'Salinity', 
            'PRES': 'Pressure',
            'temperature': 'Temperature',
            'salinity': 'Salinity',
            'pressure': 'Pressure'
        }
        
        return name_mapping.get(param, param.title())
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about conversation capabilities"""
        return {
            "description": "Generates natural language responses and manages conversation flow",
            "response_types": [rtype.value for rtype in ResponseType],
            "conversation_contexts": [ctx.value for ctx in ConversationContext],
            "supported_styles": list(self.style_guidelines.keys()),
            "features": ["executive_summary", "key_findings", "recommendations", "follow_up_suggestions"]
        }