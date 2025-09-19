"""
Data Retrieval Agent
Generates and executes optimized SQL queries from operator graphs
Handles database connections and data retrieval from PostgreSQL
"""

import json
import asyncio
import asyncpg
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_agent import BaseAgent, AgentResult, LLMMessage

class QueryType(Enum):
    """Types of database queries"""
    POINT_QUERY = "point_query"          # Single location/time
    SPATIAL_QUERY = "spatial_query"      # Geographic filtering
    TEMPORAL_QUERY = "temporal_query"    # Time-based filtering
    STATISTICAL_QUERY = "statistical_query"  # Aggregation queries
    COMPARISON_QUERY = "comparison_query"    # Cross-comparison

@dataclass
class DatabaseConnection:
    """Database connection configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str
    
    def get_connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class QueryResult:
    """Result from database query execution"""
    data: pd.DataFrame
    query: str
    execution_time: float
    row_count: int
    metadata: Dict[str, Any]

@dataclass
class QueryPlan:
    """Execution plan for database query"""
    query_type: QueryType
    sql_query: str
    parameters: Dict[str, Any]
    estimated_rows: int
    optimization_notes: List[str]

class DataRetrievalAgent(BaseAgent):
    """Agent responsible for generating SQL queries and retrieving oceanographic data"""
    
    def __init__(self, config):
        super().__init__(config, "data_retrieval")
        
        # Database connection from config
        self.db_config = DatabaseConnection(
            host=config.database_config.host,
            port=config.database_config.port,
            database=config.database_config.database,
            user=config.database_config.username,
            password=config.database_config.password
        )
        
        # Query templates for common operations
        self.query_templates = {
            'spatial_filter': """
                SELECT p.platform_id, p.latitude, p.longitude, p.cycle_number, p.profile_date,
                       m.parameter, m.pressure, m.parameter_value, m.parameter_qc
                FROM Profile p
                JOIN Measurement m ON p.profile_id = m.profile_id
                WHERE p.latitude BETWEEN {south} AND {north}
                  AND p.longitude BETWEEN {west} AND {east}
                  {time_filter}
                  {parameter_filter}
                ORDER BY p.profile_date, p.platform_id, m.pressure
            """,
            
            'temporal_aggregation': """
                SELECT 
                    DATE_TRUNC('{time_unit}', p.profile_date) as time_period,
                    m.parameter,
                    AVG(m.parameter_value) as avg_value,
                    MIN(m.parameter_value) as min_value,
                    MAX(m.parameter_value) as max_value,
                    STDDEV(m.parameter_value) as std_value,
                    COUNT(*) as measurement_count
                FROM Profile p
                JOIN Measurement m ON p.profile_id = m.profile_id
                WHERE {spatial_filter}
                  {time_filter}
                  {parameter_filter}
                  AND m.parameter_qc IN ('1', '2', '5', '8')  -- Good quality data only
                GROUP BY time_period, m.parameter
                ORDER BY time_period, m.parameter
            """,
            
            'depth_profile': """
                SELECT 
                    ROUND(m.pressure/10)*10 as depth_bin,  -- 10-meter depth bins
                    m.parameter,
                    AVG(m.parameter_value) as avg_value,
                    STDDEV(m.parameter_value) as std_value,
                    COUNT(*) as measurement_count
                FROM Profile p
                JOIN Measurement m ON p.profile_id = m.profile_id
                WHERE {spatial_filter}
                  {time_filter}
                  {parameter_filter}
                  AND m.parameter_qc IN ('1', '2', '5', '8')
                GROUP BY depth_bin, m.parameter
                ORDER BY depth_bin, m.parameter
            """,
            
            'comparison_query': """
                WITH location_data AS (
                    SELECT 
                        CASE {location_classification} END as location_group,
                        m.parameter,
                        m.parameter_value,
                        p.profile_date
                    FROM Profile p
                    JOIN Measurement m ON p.profile_id = m.profile_id
                    WHERE {spatial_filter}
                      {time_filter}
                      {parameter_filter}
                      AND m.parameter_qc IN ('1', '2', '5', '8')
                )
                SELECT 
                    location_group,
                    parameter,
                    AVG(parameter_value) as avg_value,
                    MIN(parameter_value) as min_value,
                    MAX(parameter_value) as max_value,
                    STDDEV(parameter_value) as std_value,
                    COUNT(*) as measurement_count
                FROM location_data
                GROUP BY location_group, parameter
                ORDER BY location_group, parameter
            """
        }
        
        # Connection pool
        self.connection_pool = None
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Process operator graph and execute database queries"""
        
        try:
            # Input should contain operator graph from Query Understanding Agent
            # and resolved locations from Geospatial Agent
            if not isinstance(input_data, dict):
                return AgentResult.error_result(
                    self.agent_name,
                    ["Input must be a dictionary with operator graph and location data"]
                )
            
            operator_graph = input_data.get('operator_graph')
            locations = input_data.get('locations', [])
            
            if not operator_graph:
                return AgentResult.error_result(
                    self.agent_name,
                    ["Missing operator_graph in input data"]
                )
            
            # Initialize database connection
            await self._init_db_connection()
            
            # Generate query plan from operator graph
            query_plan = await self._generate_query_plan(operator_graph, locations)
            
            # Execute query
            query_result = await self._execute_query(query_plan)
            
            return AgentResult.success_result(
                self.agent_name,
                {
                    'query_plan': self._query_plan_to_dict(query_plan),
                    'data': query_result.data.to_dict('records'),
                    'metadata': {
                        'row_count': query_result.row_count,
                        'execution_time': query_result.execution_time,
                        'columns': list(query_result.data.columns),
                        'query': query_result.query
                    }
                },
                {'rows_retrieved': query_result.row_count}
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving data: {str(e)}")
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to retrieve data: {str(e)}"]
            )
        finally:
            await self._close_db_connection()
    
    async def _init_db_connection(self):
        """Initialize database connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.db_config.get_connection_string(),
                min_size=1,
                max_size=5,
                command_timeout=60
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    async def _close_db_connection(self):
        """Close database connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            self.logger.info("Database connection pool closed")
    
    async def _generate_query_plan(self, operator_graph: Dict[str, Any], locations: List[Dict[str, Any]]) -> QueryPlan:
        """Generate SQL query plan from operator graph"""
        
        print(operator_graph)
        
        nodes = operator_graph.get('nodes', [])
        
        # Find data retrieval node (should be first in the graph)
        retrieval_node = None
        for node in nodes:
            if node['operation'] == 'data_retrieval':
                retrieval_node = node
                break
        
        if not retrieval_node:
            raise ValueError("No data_retrieval node found in operator graph")
        
        # Determine query type based on subsequent operations
        query_type = self._determine_query_type(nodes)
        
        # Build SQL query based on type
        if query_type == QueryType.SPATIAL_QUERY:
            sql_query = await self._build_spatial_query(retrieval_node, locations)
        elif query_type == QueryType.TEMPORAL_QUERY:
            sql_query = await self._build_temporal_query(retrieval_node, locations)
        elif query_type == QueryType.STATISTICAL_QUERY:
            sql_query = await self._build_statistical_query(retrieval_node, locations, nodes)
        elif query_type == QueryType.COMPARISON_QUERY:
            sql_query = await self._build_comparison_query(retrieval_node, locations, nodes)
        else:
            sql_query = await self._build_point_query(retrieval_node, locations)
        
        # Estimate result size (simplified)
        estimated_rows = await self._estimate_result_size(sql_query)
        
        return QueryPlan(
            query_type=query_type,
            sql_query=sql_query,
            parameters=retrieval_node.get('parameters', {}),
            estimated_rows=estimated_rows,
            optimization_notes=[]
        )
    
    def _determine_query_type(self, nodes: List[Dict[str, Any]]) -> QueryType:
        """Determine the type of query based on operator graph"""
        
        operations = [node['operation'] for node in nodes]
        
        if 'comparison' in operations:
            return QueryType.COMPARISON_QUERY
        elif 'statistical_analysis' in operations:
            return QueryType.STATISTICAL_QUERY
        elif any('temporal' in op for op in operations):
            return QueryType.TEMPORAL_QUERY
        elif any('spatial' in op for op in operations):
            return QueryType.SPATIAL_QUERY
        else:
            return QueryType.POINT_QUERY
    
    async def _build_spatial_query(self, node: Dict[str, Any], locations: List[Dict[str, Any]]) -> str:
        """Build spatial filter query"""
        
        parameters = node.get('parameters', {})
        
        # Build spatial filter
        spatial_conditions = []
        for location in locations:
            bounds = location['bounds']
            spatial_conditions.append(
                f"(p.latitude BETWEEN {bounds['south']} AND {bounds['north']} "
                f"AND p.longitude BETWEEN {bounds['west']} AND {bounds['east']})"
            )
        
        spatial_filter = " OR ".join(spatial_conditions) if spatial_conditions else "TRUE"
        
        # Build parameter filter
        parameter_filter = self._build_parameter_filter(parameters.get('parameters', []))
        
        # Build time filter
        time_filter = self._build_time_filter(parameters.get('time_range'))
        
        return self.query_templates['spatial_filter'].format(
            south=min(loc['bounds']['south'] for loc in locations) if locations else -90,
            north=max(loc['bounds']['north'] for loc in locations) if locations else 90,
            west=min(loc['bounds']['west'] for loc in locations) if locations else -180,
            east=max(loc['bounds']['east'] for loc in locations) if locations else 180,
            time_filter=f"AND {time_filter}" if time_filter else "",
            parameter_filter=f"AND {parameter_filter}" if parameter_filter else ""
        )
    
    async def _build_statistical_query(self, node: Dict[str, Any], locations: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> str:
        """Build statistical aggregation query"""
        
        parameters = node.get('parameters', {})
        
        # Find statistical operations
        stats_node = next((n for n in nodes if n['operation'] == 'statistical_analysis'), None)
        operations = stats_node.get('parameters', {}).get('operations', []) if stats_node else []
        
        # Determine aggregation level (temporal vs depth)
        if any('trend' in op or 'temporal' in op for op in operations):
            time_unit = 'month'  # Default to monthly aggregation
            return self._build_temporal_aggregation_query(parameters, locations, time_unit)
        else:
            return self._build_depth_profile_query(parameters, locations)
    
    async def _build_comparison_query(self, node: Dict[str, Any], locations: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> str:
        """Build comparison query across locations/parameters"""
        
        parameters = node.get('parameters', {})
        
        # Build location classification for comparison
        location_cases = []
        for i, location in enumerate(locations):
            bounds = location['bounds']
            location_cases.append(
                f"WHEN (p.latitude BETWEEN {bounds['south']} AND {bounds['north']} "
                f"AND p.longitude BETWEEN {bounds['west']} AND {bounds['east']}) THEN '{location['name']}'"
            )
        
        location_classification = "\n".join(location_cases) + "\nELSE 'Unknown'"
        
        # Build filters
        spatial_filter = "TRUE"  # We classify in the CASE statement
        time_filter = self._build_time_filter(parameters.get('time_range'))
        parameter_filter = self._build_parameter_filter(parameters.get('parameters', []))
        
        return self.query_templates['comparison_query'].format(
            location_classification=location_classification,
            spatial_filter=spatial_filter,
            time_filter=f"AND {time_filter}" if time_filter else "",
            parameter_filter=f"AND {parameter_filter}" if parameter_filter else ""
        )
    
    async def _build_point_query(self, node: Dict[str, Any], locations: List[Dict[str, Any]]) -> str:
        """Build simple point query"""
        return await self._build_spatial_query(node, locations)
    
    def _build_temporal_aggregation_query(self, parameters: Dict[str, Any], locations: List[Dict[str, Any]], time_unit: str) -> str:
        """Build temporal aggregation query"""
        
        # Build spatial filter
        spatial_conditions = []
        for location in locations:
            bounds = location['bounds']
            spatial_conditions.append(
                f"(p.latitude BETWEEN {bounds['south']} AND {bounds['north']} "
                f"AND p.longitude BETWEEN {bounds['west']} AND {bounds['east']})"
            )
        
        spatial_filter = " OR ".join(spatial_conditions) if spatial_conditions else "TRUE"
        
        # Build other filters
        time_filter = self._build_time_filter(parameters.get('time_range'))
        parameter_filter = self._build_parameter_filter(parameters.get('parameters', []))
        
        return self.query_templates['temporal_aggregation'].format(
            time_unit=time_unit,
            spatial_filter=spatial_filter,
            time_filter=f"AND {time_filter}" if time_filter else "",
            parameter_filter=f"AND {parameter_filter}" if parameter_filter else ""
        )
    
    def _build_depth_profile_query(self, parameters: Dict[str, Any], locations: List[Dict[str, Any]]) -> str:
        """Build depth profile query"""
        
        spatial_conditions = []
        for location in locations:
            bounds = location['bounds']
            spatial_conditions.append(
                f"(p.latitude BETWEEN {bounds['south']} AND {bounds['north']} "
                f"AND p.longitude BETWEEN {bounds['west']} AND {bounds['east']})"
            )
        
        spatial_filter = " OR ".join(spatial_conditions) if spatial_conditions else "TRUE"
        time_filter = self._build_time_filter(parameters.get('time_range'))
        parameter_filter = self._build_parameter_filter(parameters.get('parameters', []))
        
        return self.query_templates['depth_profile'].format(
            spatial_filter=spatial_filter,
            time_filter=f"AND {time_filter}" if time_filter else "",
            parameter_filter=f"AND {parameter_filter}" if parameter_filter else ""
        )
    
    def _build_parameter_filter(self, parameters: List[str]) -> str:
        """Build parameter filter clause"""
        if not parameters:
            return ""
        
        # Map common parameter names to database values
        param_mapping = {
            'temperature': 'TEMP',
            'temp': 'TEMP',
            'salinity': 'PSAL',
            'psal': 'PSAL',
            'pressure': 'PRES',
            'pres': 'PRES'
        }
        
        db_params = []
        for param in parameters:
            if param.lower() in param_mapping:
                db_params.append(f"'{param_mapping[param.lower()]}'")
        
        if db_params:
            return f"m.parameter IN ({', '.join(db_params)})"
        else:
            return ""
    
    def _build_time_filter(self, time_range: Optional[Dict[str, Any]]) -> str:
        """Build time filter clause"""
        if not time_range:
            return ""
        
        time_type = time_range.get('type', 'point')
        
        if time_type == 'range' and time_range.get('start') and time_range.get('end'):
            return f"p.profile_date BETWEEN '{time_range['start']}' AND '{time_range['end']}'"
        elif time_type == 'recent':
            # Default to last 30 days
            return f"p.profile_date >= CURRENT_DATE - INTERVAL '30 days'"
        elif time_type == 'seasonal':
            # Extract month for seasonal analysis
            description = time_range.get('description', '').lower()
            if 'winter' in description:
                return "EXTRACT(MONTH FROM p.profile_date) IN (12, 1, 2)"
            elif 'spring' in description:
                return "EXTRACT(MONTH FROM p.profile_date) IN (3, 4, 5)"
            elif 'summer' in description:
                return "EXTRACT(MONTH FROM p.profile_date) IN (6, 7, 8)"
            elif 'fall' in description or 'autumn' in description:
                return "EXTRACT(MONTH FROM p.profile_date) IN (9, 10, 11)"
        
        return ""
    
    async def _estimate_result_size(self, query: str) -> int:
        """Estimate the number of rows the query will return"""
        # Simplified estimation - could be improved with EXPLAIN
        if "COUNT(*)" in query.upper():
            return 1000  # Aggregation queries return fewer rows
        elif "GROUP BY" in query.upper():
            return 5000  # Grouped queries
        else:
            return 50000  # Raw data queries
    
    async def _execute_query(self, query_plan: QueryPlan) -> QueryResult:
        """Execute the SQL query and return results"""
        
        start_time = asyncio.get_event_loop().time()
        
        async with self.connection_pool.acquire() as connection:
            try:
                # Execute query
                rows = await connection.fetch(query_plan.sql_query)
                
                # Convert to pandas DataFrame
                if rows:
                    data = pd.DataFrame([dict(row) for row in rows])
                else:
                    data = pd.DataFrame()
                
                execution_time = asyncio.get_event_loop().time() - start_time
                
                self.logger.info(f"Query executed successfully: {len(data)} rows in {execution_time:.2f}s")
                
                return QueryResult(
                    data=data,
                    query=query_plan.sql_query,
                    execution_time=execution_time,
                    row_count=len(data),
                    metadata={
                        'query_type': query_plan.query_type.value,
                        'estimated_rows': query_plan.estimated_rows
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise
    
    def _query_plan_to_dict(self, query_plan: QueryPlan) -> Dict[str, Any]:
        """Convert QueryPlan to dictionary"""
        return {
            'query_type': query_plan.query_type.value,
            'estimated_rows': query_plan.estimated_rows,
            'optimization_notes': query_plan.optimization_notes,
            'parameters': query_plan.parameters
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about data retrieval capabilities"""
        return {
            "description": "Generates and executes SQL queries for oceanographic data retrieval",
            "supported_operations": ["spatial_filter", "temporal_aggregation", "statistical_analysis", "comparison"],
            "query_types": [qtype.value for qtype in QueryType],
            "parameters": ["TEMP", "PSAL", "PRES"],
            "quality_control": "Filters for good quality data (QC flags 1,2,5,8)"
        }