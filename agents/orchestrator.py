"""
Agent Orchestrator
Coordinates workflow between all agents, manages execution order,
and handles inter-agent communication for oceanographic data queries
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

from .base_agent import BaseAgent, AgentResult
from .query_understanding import QueryUnderstandingAgent
from .geospatial import GeospatialAgent  
from .data_retrieval import DataRetrievalAgent
from .analysis import AnalysisAgent
from .visualization import VisualizationAgent
from .critic import CriticAgent
from .conversation import ConversationAgent

class ExecutionStrategy(Enum):
    """Strategies for agent execution"""
    SEQUENTIAL = "sequential"         # Execute agents in order
    PARALLEL = "parallel"            # Execute independent agents in parallel
    ADAPTIVE = "adaptive"            # Adapt based on query complexity
    FAIL_FAST = "fail_fast"         # Stop on first error
    BEST_EFFORT = "best_effort"     # Continue despite errors

class WorkflowStage(Enum):
    """Stages of the analysis workflow"""
    UNDERSTANDING = "understanding"   # Query understanding and planning
    PREPARATION = "preparation"       # Data location and retrieval preparation
    RETRIEVAL = "retrieval"          # Data retrieval and initial processing
    ANALYSIS = "analysis"            # Statistical analysis and computation
    SYNTHESIS = "synthesis"          # Visualization and validation
    RESPONSE = "response"            # Final response generation

@dataclass
class ExecutionPlan:
    """Plan for executing agents to fulfill a query"""
    stages: List[WorkflowStage]
    agent_assignments: Dict[WorkflowStage, List[str]]
    dependencies: Dict[str, List[str]]
    strategy: ExecutionStrategy
    estimated_duration: float

@dataclass
class ExecutionContext:
    """Context shared between agents during execution"""
    session_id: str
    original_query: str
    user_preferences: Dict[str, Any]
    intermediate_results: Dict[str, AgentResult]
    execution_metadata: Dict[str, Any]
    start_time: datetime

class AgentOrchestrator(BaseAgent):
    """Orchestrator that coordinates all agents to fulfill user queries"""
    
    def __init__(self, config):
        super().__init__(config, "orchestrator")
        
        # Initialize all agents
        self.agents = {
            'query_understanding': QueryUnderstandingAgent(config),
            'geospatial': GeospatialAgent(config),
            'data_retrieval': DataRetrievalAgent(config),
            'analysis': AnalysisAgent(config),
            'visualization': VisualizationAgent(config),
            'critic': CriticAgent(config),
            'conversation': ConversationAgent(config)
        }
        
        # Default workflow configuration
        self.default_workflow = {
            WorkflowStage.UNDERSTANDING: ['query_understanding'],
            WorkflowStage.PREPARATION: ['geospatial'],
            WorkflowStage.RETRIEVAL: ['data_retrieval'],
            WorkflowStage.ANALYSIS: ['analysis'],
            WorkflowStage.SYNTHESIS: ['visualization', 'critic'],
            WorkflowStage.RESPONSE: ['conversation']
        }
        
        # Agent dependencies
        self.dependencies = {
            'geospatial': ['query_understanding'],
            'data_retrieval': ['query_understanding', 'geospatial'],
            'analysis': ['data_retrieval'],
            'visualization': ['analysis'],
            'critic': ['data_retrieval', 'analysis'],
            'conversation': ['query_understanding', 'analysis', 'visualization', 'critic']
        }
        
        # Execution settings
        self.max_retries = 2
        self.timeout_per_agent = 300  # 5 minutes per agent
        self.parallel_limit = 3       # Max parallel agents
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Process a user query through the complete agent workflow"""
        
        try:
            if not isinstance(input_data, str):
                return AgentResult.error_result(
                    self.agent_name,
                    ["Input must be a string query"]
                )
            
            user_query = input_data.strip()
            session_id = context.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            self.logger.info(f"Starting orchestration for query: {user_query[:100]}...")
            
            # Create execution context
            execution_context = ExecutionContext(
                session_id=session_id,
                original_query=user_query,
                user_preferences=context.get('user_preferences', {}),
                intermediate_results={},
                execution_metadata={'stage_timings': {}},
                start_time=datetime.now()
            )
            
            # Generate execution plan
            execution_plan = await self._create_execution_plan(user_query, context or {})
            
            # Execute workflow
            final_result = await self._execute_workflow(execution_plan, execution_context)
            
            # Calculate total execution time
            total_time = (datetime.now() - execution_context.start_time).total_seconds()
            
            return AgentResult.success_result(
                self.agent_name,
                {
                    'response': final_result.get('conversation', {}).get('data', {}).get('response', 'Analysis completed successfully.'),
                    'visualizations': final_result.get('visualization', {}).get('data', {}).get('visualizations', {}),
                    'validation_report': final_result.get('critic', {}).get('data', {}).get('validation_report', {}),
                    'execution_summary': {
                        'total_time': total_time,
                        'agents_executed': list(final_result.keys()),
                        'stages_completed': len(execution_plan.stages),
                        'session_id': session_id
                    },
                    'intermediate_results': {k: v.data for k, v in execution_context.intermediate_results.items() if v.success}
                },
                {'execution_time': total_time}
            )
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {str(e)}")
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to process query: {str(e)}"]
            )
    
    async def _create_execution_plan(self, query: str, context: Dict[str, Any]) -> ExecutionPlan:
        """Create execution plan based on query complexity and requirements"""
        
        # Analyze query complexity (simple heuristics)
        query_lower = query.lower()
        
        # Determine execution strategy
        if any(word in query_lower for word in ['compare', 'correlation', 'trend', 'anomaly']):
            strategy = ExecutionStrategy.SEQUENTIAL  # Complex analysis needs sequential execution
            stages = list(WorkflowStage)
        elif any(word in query_lower for word in ['map', 'show', 'visualize']):
            strategy = ExecutionStrategy.ADAPTIVE
            stages = [WorkflowStage.UNDERSTANDING, WorkflowStage.PREPARATION, 
                     WorkflowStage.RETRIEVAL, WorkflowStage.SYNTHESIS, WorkflowStage.RESPONSE]
        else:
            strategy = ExecutionStrategy.BEST_EFFORT
            stages = list(WorkflowStage)
        
        # Assign agents to stages
        agent_assignments = {}
        for stage in stages:
            if stage in self.default_workflow:
                agent_assignments[stage] = self.default_workflow[stage]
        
        return ExecutionPlan(
            stages=stages,
            agent_assignments=agent_assignments,
            dependencies=self.dependencies,
            strategy=strategy,
            estimated_duration=len(stages) * 30  # Rough estimate: 30s per stage
        )
    
    async def _execute_workflow(self, plan: ExecutionPlan, context: ExecutionContext) -> Dict[str, AgentResult]:
        """Execute the complete workflow according to the plan"""
        
        results = {}
        
        for stage in plan.stages:
            stage_start = datetime.now()
            self.logger.info(f"Executing stage: {stage.value}")
            
            try:
                if stage not in plan.agent_assignments:
                    continue
                
                agents_for_stage = plan.agent_assignments[stage]
                
                # Execute agents in this stage
                if plan.strategy == ExecutionStrategy.PARALLEL and len(agents_for_stage) > 1:
                    stage_results = await self._execute_agents_parallel(
                        agents_for_stage, context, results
                    )
                else:
                    stage_results = await self._execute_agents_sequential(
                        agents_for_stage, context, results
                    )
                
                # Update results
                results.update(stage_results)
                
                # Record stage timing
                stage_duration = (datetime.now() - stage_start).total_seconds()
                context.execution_metadata['stage_timings'][stage.value] = stage_duration
                
                # Check if we should continue
                if plan.strategy == ExecutionStrategy.FAIL_FAST:
                    failed_agents = [name for name, result in stage_results.items() if not result.success]
                    if failed_agents:
                        self.logger.warning(f"Stopping execution due to failures: {failed_agents}")
                        break
                
            except Exception as e:
                self.logger.error(f"Stage {stage.value} failed: {e}")
                if plan.strategy == ExecutionStrategy.FAIL_FAST:
                    break
                continue
        
        return results
    
    async def _execute_agents_sequential(self, agent_names: List[str], 
                                       context: ExecutionContext,
                                       previous_results: Dict[str, AgentResult]) -> Dict[str, AgentResult]:
        """Execute agents sequentially"""
        
        results = {}
        
        for agent_name in agent_names:
            if agent_name not in self.agents:
                self.logger.warning(f"Agent {agent_name} not found")
                continue
            
            try:
                # Prepare input data for agent
                agent_input = await self._prepare_agent_input(agent_name, context, previous_results)
                agent_context = await self._prepare_agent_context(agent_name, context, previous_results)
                
                # Log agent input data
                self.logger.info(f"=== AGENT INPUT for {agent_name} ===")
                self.logger.info(f"Input Data Type: {type(agent_input)}")
                self.logger.info(f"Input Data: {agent_input}")
                self.logger.info(f"Context: {agent_context}")
                self.logger.info(f"=== END AGENT INPUT ===")
                
                # Execute agent with timeout
                self.logger.info(f"Executing agent: {agent_name}")
                result = await asyncio.wait_for(
                    self.agents[agent_name].process(agent_input, agent_context),
                    timeout=self.timeout_per_agent
                )
                
                # Log agent output data
                self.logger.info(f"=== AGENT OUTPUT for {agent_name} ===")
                self.logger.info(f"Success: {result.success}")
                self.logger.info(f"Data Type: {type(result.data)}")
                self.logger.info(f"Data: {result.data}")
                self.logger.info(f"Metadata: {result.metadata}")
                if result.errors:
                    self.logger.info(f"Errors: {result.errors}")
                self.logger.info(f"=== END AGENT OUTPUT ===")
                
                results[agent_name] = result
                context.intermediate_results[agent_name] = result
                
                if result.success:
                    self.logger.info(f"Agent {agent_name} completed successfully")
                else:
                    self.logger.warning(f"Agent {agent_name} failed: {result.errors}")
                
            except asyncio.TimeoutError:
                error_msg = f"Agent {agent_name} timed out after {self.timeout_per_agent}s"
                self.logger.error(error_msg)
                results[agent_name] = AgentResult.error_result(agent_name, [error_msg])
                
            except Exception as e:
                error_msg = f"Agent {agent_name} execution failed: {str(e)}"
                self.logger.error(error_msg)
                results[agent_name] = AgentResult.error_result(agent_name, [error_msg])
        
        return results
    
    async def _execute_agents_parallel(self, agent_names: List[str],
                                     context: ExecutionContext,
                                     previous_results: Dict[str, AgentResult]) -> Dict[str, AgentResult]:
        """Execute independent agents in parallel"""
        
        # Filter agents that can run in parallel (no dependencies on each other)
        executable_agents = []
        for agent_name in agent_names:
            if agent_name in self.agents:
                # Check if dependencies are satisfied
                deps = self.dependencies.get(agent_name, [])
                if all(dep in previous_results and previous_results[dep].success for dep in deps):
                    executable_agents.append(agent_name)
        
        if not executable_agents:
            return {}
        
        # Limit concurrent execution
        semaphore = asyncio.Semaphore(min(self.parallel_limit, len(executable_agents)))
        
        async def execute_single_agent(agent_name: str) -> Tuple[str, AgentResult]:
            async with semaphore:
                try:
                    agent_input = await self._prepare_agent_input(agent_name, context, previous_results)
                    agent_context = await self._prepare_agent_context(agent_name, context, previous_results)
                    
                    self.logger.info(f"Executing agent (parallel): {agent_name}")
                    result = await asyncio.wait_for(
                        self.agents[agent_name].process(agent_input, agent_context),
                        timeout=self.timeout_per_agent
                    )
                    
                    return agent_name, result
                    
                except asyncio.TimeoutError:
                    error_msg = f"Agent {agent_name} timed out after {self.timeout_per_agent}s"
                    return agent_name, AgentResult.error_result(agent_name, [error_msg])
                except Exception as e:
                    error_msg = f"Agent {agent_name} execution failed: {str(e)}"
                    return agent_name, AgentResult.error_result(agent_name, [error_msg])
        
        # Execute agents in parallel
        tasks = [execute_single_agent(agent_name) for agent_name in executable_agents]
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {}
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                self.logger.error(f"Parallel execution error: {task_result}")
                continue
            
            agent_name, result = task_result
            results[agent_name] = result
            context.intermediate_results[agent_name] = result
            
            if result.success:
                self.logger.info(f"Agent {agent_name} completed successfully (parallel)")
            else:
                self.logger.warning(f"Agent {agent_name} failed (parallel): {result.errors}")
        
        return results
    
    async def _prepare_agent_input(self, agent_name: str, context: ExecutionContext,
                                 previous_results: Dict[str, AgentResult]) -> Any:
        """Prepare input data for specific agent based on its requirements"""
        
        # Query Understanding Agent - gets raw query
        if agent_name == 'query_understanding':
            return context.original_query
        
        # Geospatial Agent - gets locations from query understanding
        elif agent_name == 'geospatial':
            if 'query_understanding' in previous_results:
                entities = previous_results['query_understanding'].data.get('entities')
                if entities and hasattr(entities, 'locations'):
                    return entities.locations
            return []
        
        # Data Retrieval Agent - gets operator graph and locations
        elif agent_name == 'data_retrieval':
            input_data = {}
            if 'query_understanding' in previous_results:
                input_data['operator_graph'] = previous_results['query_understanding'].data.get('operator_graph', {})
            if 'geospatial' in previous_results:
                input_data['locations'] = previous_results['geospatial'].data.get('locations', [])
            return input_data
        
        # Analysis Agent - gets retrieved data
        elif agent_name == 'analysis':
            if 'data_retrieval' in previous_results:
                return previous_results['data_retrieval'].data
            return {}
        
        # Visualization Agent - gets data and analysis results
        elif agent_name == 'visualization':
            input_data = {}
            if 'data_retrieval' in previous_results:
                input_data.update(previous_results['data_retrieval'].data)
            if 'analysis' in previous_results:
                input_data['analysis_results'] = previous_results['analysis'].data
            return input_data
        
        # Critic Agent - gets all data and analysis results  
        elif agent_name == 'critic':
            input_data = {}
            if 'data_retrieval' in previous_results:
                input_data.update(previous_results['data_retrieval'].data)
            if 'analysis' in previous_results:
                input_data['analysis_results'] = previous_results['analysis'].data
            if 'visualization' in previous_results:
                input_data['visualizations'] = previous_results['visualization'].data
            return input_data
        
        # Conversation Agent - gets all results
        elif agent_name == 'conversation':
            print("mayur find the error here")
            print("*"*20)
            import pprint
            
            pprint.pprint(previous_results)
            
            print("*"*20)
            print("mayur end")
            return {
                'query_understanding': previous_results.get('query_understanding', {}).data,
                'geospatial': previous_results.get('geospatial', {}).data,
                'data_retrieval': previous_results.get('data_retrieval', {}).data,
                'analysis': previous_results.get('analysis', {}).data,
                'visualization': previous_results.get('visualization', {}).data,
                'validation': previous_results.get('critic', {}).data
            }
        
        return context.original_query  # Fallback
    
    async def _prepare_agent_context(self, agent_name: str, context: ExecutionContext,
                                   previous_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Prepare context for specific agent"""
        
        agent_context = {
            'session_id': context.session_id,
            'user_preferences': context.user_preferences,
            'execution_metadata': context.execution_metadata
        }
        
        # Add relevant previous results to context
        if agent_name in ['analysis', 'visualization', 'critic', 'conversation']:
            if 'query_understanding' in previous_results:
                agent_context['operator_graph'] = previous_results['query_understanding'].data.get('operator_graph', {})
                entities = previous_results['query_understanding'].data.get('entities')
                agent_context['entities'] = entities if entities else {}
        
        return agent_context
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about orchestrator capabilities"""
        
        agent_capabilities = {}
        for name, agent in self.agents.items():
            try:
                agent_capabilities[name] = agent.get_capabilities()
            except Exception as e:
                agent_capabilities[name] = {'error': f'Failed to get capabilities: {e}'}
        
        return {
            "description": "Orchestrates oceanographic data analysis through coordinated agent workflow",
            "agents": list(self.agents.keys()),
            "workflow_stages": [stage.value for stage in WorkflowStage],
            "execution_strategies": [strategy.value for strategy in ExecutionStrategy],
            "agent_capabilities": agent_capabilities,
            "max_parallel_agents": self.parallel_limit,
            "timeout_per_agent": self.timeout_per_agent
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        
        health_status = {
            'orchestrator': 'healthy',
            'agents': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check each agent
        for name, agent in self.agents.items():
            try:
                agent_health = await agent.health_check()
                health_status['agents'][name] = agent_health
            except Exception as e:
                health_status['agents'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Overall health
        unhealthy_agents = [name for name, status in health_status['agents'].items() 
                           if isinstance(status, dict) and status.get('status') != 'healthy']
        
        if unhealthy_agents:
            health_status['orchestrator'] = 'degraded'
            health_status['unhealthy_agents'] = unhealthy_agents
        
        return health_status