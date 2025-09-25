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

# Add query-specific logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.query_logging import get_query_logger

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
        
        # Query-specific logging manager
        self.query_logger_manager = get_query_logger()
        self.current_query_logger = None
        
        # Query templates for common operations
        self.query_templates = {
            'spatial_filter': """
                SELECT p.platform_id, p.latitude, p.longitude, p.cycle_number, p.profile_date,
                       m.pressure, m.temp as temperature, m.psal as salinity,
                       m.temp_qc as temperature_qc, m.psal_qc as salinity_qc, m.pressure_qc
                FROM profiles p
                JOIN measurements m ON p.id = m.profile_id
                WHERE p.latitude BETWEEN {south} AND {north}
                  AND p.longitude BETWEEN {west} AND {east}
                  {time_filter}
                  {parameter_filter}
                ORDER BY p.profile_date, p.platform_id, m.pressure
            """,
            
            'temporal_aggregation': """
                SELECT 
                    DATE_TRUNC('{time_unit}', p.profile_date) as time_period,
                    '{parameter}' as parameter,
                    AVG({parameter_column}) as avg_value,
                    MIN({parameter_column}) as min_value,
                    MAX({parameter_column}) as max_value,
                    STDDEV({parameter_column}) as std_value,
                    COUNT(*) as measurement_count
                FROM profiles p
                JOIN measurements m ON p.id = m.profile_id
                WHERE {spatial_filter}
                  {time_filter}
                  AND {parameter_column} IS NOT NULL
                  AND {qc_filter}
                GROUP BY time_period
                ORDER BY time_period
            """,
            
            'depth_profile': """
                SELECT 
                    ROUND(m.pressure/10)*10 as depth_bin,  -- 10-meter depth bins
                    '{parameter}' as parameter,
                    AVG({parameter_column}) as avg_value,
                    STDDEV({parameter_column}) as std_value,
                    COUNT(*) as measurement_count
                FROM profiles p
                JOIN measurements m ON p.id = m.profile_id
                WHERE {spatial_filter}
                  {time_filter}
                  AND {parameter_column} IS NOT NULL
                  AND {qc_filter}
                GROUP BY depth_bin
                ORDER BY depth_bin
            """,
            
            'comparison_query': """
                WITH location_data AS (
                    SELECT 
                        CASE {location_classification} END as location_group,
                        '{parameter}' as parameter,
                        {parameter_column} as parameter_value,
                        p.profile_date
                    FROM profiles p
                    JOIN measurements m ON p.id = m.profile_id
                    WHERE {spatial_filter}
                      {time_filter}
                      AND {parameter_column} IS NOT NULL
                      AND {qc_filter}
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
    
    def _get_column_mapping(self, parameter: str) -> Dict[str, str]:
        """Get database column names for a given parameter"""
        mapping = {
            'temperature': {
                'column': 'temp',
                'qc_column': 'temp_qc',
                'adjusted_column': 'temp_adjusted'
            },
            'temp': {
                'column': 'temp', 
                'qc_column': 'temp_qc',
                'adjusted_column': 'temp_adjusted'
            },
            'salinity': {
                'column': 'psal',
                'qc_column': 'psal_qc', 
                'adjusted_column': 'psal_adjusted'
            },
            'psal': {
                'column': 'psal',
                'qc_column': 'psal_qc',
                'adjusted_column': 'psal_adjusted'
            },
            'pressure': {
                'column': 'pressure',
                'qc_column': 'pressure_qc',
                'adjusted_column': 'pressure_adjusted'
            },
            'pres': {
                'column': 'pressure',
                'qc_column': 'pressure_qc', 
                'adjusted_column': 'pressure_adjusted'
            }
        }
        return mapping.get(parameter.lower(), {})
    
    def _build_qc_filter(self, parameter: str) -> str:
        """Build quality control filter for parameter"""
        column_info = self._get_column_mapping(parameter)
        qc_column = column_info.get('qc_column')
        if qc_column:
            return f"({qc_column} IN (1, 2, 5, 8) OR {qc_column} IS NULL)"
        return "TRUE"
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Process operator graph and execute database queries"""
        
        # Use shared query logger from context, or create new one as fallback
        if context and 'current_query_logger' in context:
            self.current_query_logger = context['current_query_logger']
        else:
            # Fallback: create new logger (for backward compatibility)
            original_query = context.get('original_query', 'unknown_query') if context else 'unknown_query'
            session_id = context.get('session_id', None) if context else None
            self.current_query_logger = self.query_logger_manager.get_query_logger(original_query, session_id)
        
        try:
            # Log agent start
            self.current_query_logger.log_agent_start(self.agent_name, input_data)
            
            # Input should contain operator graph from Query Understanding Agent
            # and resolved locations from Geospatial Agent
            if not isinstance(input_data, dict):
                error_msg = "Input must be a dictionary with operator graph and location data"
                self.current_query_logger.log_error(
                    self.agent_name, 
                    ValueError(error_msg), "Input validation"
                )
                return AgentResult.error_result(self.agent_name, [error_msg])
            
            operator_graph = input_data.get('operator_graph')
            locations = input_data.get('locations', [])
            
            self.current_query_logger.info(f"Operator graph type: {type(operator_graph)}")
            self.current_query_logger.info(f"Number of locations: {len(locations)}")
            
            if not operator_graph:
                error_msg = "Missing operator_graph in input data"
                self.current_query_logger.log_error(
                    self.agent_name,
                    ValueError(error_msg), "Input validation"
                )
                return AgentResult.error_result(self.agent_name, [error_msg])
            
            # Initialize database connection
            self.current_query_logger.info("🔄 Initializing database connection...")
            await self._init_db_connection()
            self.current_query_logger.info("✅ Database connection established")
            
            # Generate query plan from operator graph
            self.current_query_logger.info("📋 Generating query plan from operator graph...")
            query_plan = await self._generate_query_plan(operator_graph, locations)
            self.current_query_logger.info(f"✅ Query plan generated: {query_plan.query_type.value}")
            
            # Log the SQL query
            self.current_query_logger.log_sql_query(
                query_plan.sql_query, 
                query_plan.parameters
            )
            
            # Execute query
            self.current_query_logger.info("⚡ Executing database query...")
            query_result = await self._execute_query(query_plan)
            
            # Log database results
            self.current_query_logger.log_database_result(
                query_result.row_count,
                query_result.execution_time,
                list(query_result.data.columns)
            )
            
            result = AgentResult.success_result(
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
            
            # Log successful result
            self.current_query_logger.log_result(self.agent_name, result)
            return result
            
        except Exception as e:
            # Log error with full context
            self.current_query_logger.log_error(
                self.agent_name, e, "Query processing"
            )
            self.logger.error(f"Error retrieving data: {str(e)}")
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to retrieve data: {str(e)}"]
            )
        finally:
            # Clean up database connection
            await self._close_db_connection()
            
            # Don't close shared query logger - it will be closed by orchestrator
            self.current_query_logger = None
    
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
    
    async def _generate_query_plan(self, operator_graph, locations: List[Dict[str, Any]]) -> QueryPlan:
        """Generate SQL query plan from operator graph"""
        
        self.logger.info(f"Operator graph type: {type(operator_graph)}")
        self.logger.info(f"Operator graph: {operator_graph}")
        
        # Handle both dict and OperatorGraph object
        if hasattr(operator_graph, 'nodes'):
            # It's an OperatorGraph dataclass
            nodes = operator_graph.nodes
        elif isinstance(operator_graph, dict):
            # It's a dictionary
            nodes = operator_graph.get('nodes', [])
        else:
            raise ValueError(f"Unexpected operator_graph type: {type(operator_graph)}")
        
        # Find data retrieval node (should be first in the graph)
        retrieval_node = None
        for node in nodes:
            # Handle both dict nodes and OperatorNode objects
            if hasattr(node, 'operation'):
                operation = node.operation
            elif isinstance(node, dict):
                operation = node.get('operation')
            else:
                continue
                
            if operation == 'data_retrieval':
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
            parameters=self._get_node_parameters(retrieval_node),
            estimated_rows=estimated_rows,
            optimization_notes=[]
        )
    
    def _determine_query_type(self, nodes) -> QueryType:
        """Determine the type of query based on operator graph"""
        
        operations = []
        for node in nodes:
            if hasattr(node, 'operation'):
                operations.append(node.operation)
            elif isinstance(node, dict):
                operations.append(node.get('operation', ''))
        
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
    
    def _get_node_parameters(self, node) -> Dict[str, Any]:
        """Get parameters from node (handles both dict and OperatorNode object)"""
        if hasattr(node, 'parameters'):
            return node.parameters
        elif isinstance(node, dict):
            return node.get('parameters', {})
        else:
            return {}
    
    async def _build_spatial_query(self, node, locations: List[Dict[str, Any]]) -> str:
        """Build spatial filter query"""
        
        parameters = self._get_node_parameters(node)
        requested_params = parameters.get('parameters', ['temperature'])  # Default to temperature
        
        # Build spatial filter
        spatial_conditions = []
        for location in locations:
            bounds = location['bounds']
            spatial_conditions.append(
                f"(p.latitude BETWEEN {bounds['south']} AND {bounds['north']} "
                f"AND p.longitude BETWEEN {bounds['west']} AND {bounds['east']})"
            )
        
        spatial_filter = " OR ".join(spatial_conditions) if spatial_conditions else "TRUE"
        
        # Build time filter
        time_filter = self._build_time_filter(parameters.get('time_range'))
        
        # For spatial queries, we'll return all available data columns
        # Use basic template but modify column selection
        query = """
            SELECT p.platform_id, p.latitude, p.longitude, p.cycle_number, p.profile_date,
                   m.pressure, m.temp as temperature, m.psal as salinity,
                   m.temp_qc as temperature_qc, m.psal_qc as salinity_qc, m.pressure_qc
            FROM profiles p
            LEFT JOIN measurements m ON p.id = m.profile_id
            WHERE ({spatial_filter})
              {time_filter}
            ORDER BY p.profile_date, p.platform_id, m.pressure
        """.format(
            spatial_filter=spatial_filter,
            time_filter=f"AND {time_filter}" if time_filter else ""
        )
        
        return query
    
    async def _build_statistical_query(self, node, locations: List[Dict[str, Any]], nodes) -> str:
        """Build statistical aggregation query"""
        
        parameters = self._get_node_parameters(node)
        requested_params = parameters.get('parameters', ['temperature'])
        
        # Find statistical operations
        stats_node = None
        for n in nodes:
            operation = n.operation if hasattr(n, 'operation') else n.get('operation', '')
            if operation == 'statistical_analysis':
                stats_node = n
                break
        
        operations = []
        if stats_node:
            stats_params = self._get_node_parameters(stats_node)
            operations = stats_params.get('operations', [])
        
        # Build spatial filter
        spatial_conditions = []
        for location in locations:
            bounds = location['bounds']
            spatial_conditions.append(
                f"(p.latitude BETWEEN {bounds['south']} AND {bounds['north']} "
                f"AND p.longitude BETWEEN {bounds['west']} AND {bounds['east']})"
            )
        
        spatial_filter = " OR ".join(spatial_conditions) if spatial_conditions else "TRUE"
        time_filter = self._build_time_filter(parameters.get('time_range'))
        
        # Determine aggregation level (temporal vs depth)
        if any('trend' in op or 'temporal' in op for op in operations):
            time_unit = 'month'  # Default to monthly aggregation
            
            # Build UNION query for all requested parameters
            param_queries = []
            for param in requested_params:
                column_info = self._get_column_mapping(param)
                if column_info:
                    time_condition = f"AND {time_filter}" if time_filter else ""
                    param_query = f"""
                        SELECT 
                            DATE_TRUNC('{time_unit}', p.profile_date) as time_period,
                            '{param}' as parameter,
                            AVG({column_info['column']}) as avg_value,
                            MIN({column_info['column']}) as min_value,
                            MAX({column_info['column']}) as max_value,
                            STDDEV({column_info['column']}) as std_value,
                            COUNT(*) as measurement_count
                        FROM profiles p
                        JOIN measurements m ON p.id = m.profile_id
                        WHERE ({spatial_filter})
                          {time_condition}
                          AND {column_info['column']} IS NOT NULL
                          AND {self._build_qc_filter(param)}
                        GROUP BY time_period
                    """
                    param_queries.append(param_query)
            
            if param_queries:
                return " UNION ALL ".join(param_queries) + " ORDER BY time_period, parameter"
            
        # Default to depth profile
        return self._build_depth_profile_query(parameters, locations)
    
    async def _build_comparison_query(self, node, locations: List[Dict[str, Any]], nodes) -> str:
        """Build comparison query across locations/parameters"""
        
        parameters = self._get_node_parameters(node)
        requested_params = parameters.get('parameters', ['temperature'])
        time_filter = self._build_time_filter(parameters.get('time_range'))
        
        # Build location-specific queries for each parameter
        all_queries = []
        
        for param in requested_params:
            column_info = self._get_column_mapping(param)
            if not column_info:
                continue
                
            location_queries = []
            for location in locations:
                bounds = location['bounds']
                location_name = location.get('name', 'Unknown')
                
                query = f"""
                    SELECT 
                        '{location_name}' as location_group,
                        '{param}' as parameter,
                        AVG({column_info['column']}) as avg_value,
                        MIN({column_info['column']}) as min_value,
                        MAX({column_info['column']}) as max_value,
                        STDDEV({column_info['column']}) as std_value,
                        COUNT(*) as measurement_count
                    FROM profiles p
                    JOIN measurements m ON p.id = m.profile_id
                    WHERE p.latitude BETWEEN {bounds['south']} AND {bounds['north']}
                      AND p.longitude BETWEEN {bounds['west']} AND {bounds['east']}
                      {f"AND {time_filter}" if time_filter else ""}
                      AND {column_info['column']} IS NOT NULL
                      AND {self._build_qc_filter(param)}
                """
                location_queries.append(query)
            
            if location_queries:
                all_queries.extend(location_queries)
        
        if all_queries:
            return " UNION ALL ".join(all_queries) + " ORDER BY location_group, parameter"
        
        # Fallback to simple spatial query
        return await self._build_spatial_query(node, locations)
    
    async def _build_point_query(self, node, locations: List[Dict[str, Any]]) -> str:
        """Build simple point query"""
        return await self._build_spatial_query(node, locations)
    
    def _build_depth_profile_query(self, parameters: Dict[str, Any], locations: List[Dict[str, Any]]) -> str:
        """Build depth profile query"""
        
        requested_params = parameters.get('parameters', ['temperature'])
        
        # Build spatial filter
        spatial_conditions = []
        for location in locations:
            bounds = location['bounds']
            spatial_conditions.append(
                f"(p.latitude BETWEEN {bounds['south']} AND {bounds['north']} "
                f"AND p.longitude BETWEEN {bounds['west']} AND {bounds['east']})"
            )
        
        spatial_filter = " OR ".join(spatial_conditions) if spatial_conditions else "TRUE"
        time_filter = self._build_time_filter(parameters.get('time_range'))
        
        # Build query for each parameter
        param_queries = []
        for param in requested_params:
            column_info = self._get_column_mapping(param)
            if column_info:
                query = f"""
                    SELECT 
                        '{param}' as parameter,
                        FLOOR(m.pressure / 10) * 10 as depth_bin,
                        AVG({column_info['column']}) as avg_value,
                        COUNT(*) as measurement_count
                    FROM profiles p
                    JOIN measurements m ON p.id = m.profile_id
                    WHERE ({spatial_filter})
                      {f"AND {time_filter}" if time_filter else ""}
                      AND {column_info['column']} IS NOT NULL
                      AND m.pressure IS NOT NULL
                      AND {self._build_qc_filter(param)}
                    GROUP BY depth_bin
                """
                param_queries.append(query)
        
        if param_queries:
            return " UNION ALL ".join(param_queries) + " ORDER BY parameter, depth_bin"
        
        # Fallback simple query
        return f"""
            SELECT 
                'temperature' as parameter,
                FLOOR(m.pressure / 10) * 10 as depth_bin,
                AVG(m.temp) as avg_value,
                COUNT(*) as measurement_count
            FROM profiles p
            JOIN measurements m ON p.id = m.profile_id
            WHERE ({spatial_filter})
              {f"AND {time_filter}" if time_filter else ""}
              AND m.temp IS NOT NULL
              AND m.pressure IS NOT NULL
              AND m.temp_qc IN (1, 2)
            GROUP BY depth_bin
            ORDER BY depth_bin
        """
    
    def _build_parameter_filter(self, parameters: List[str]) -> str:
        """Build parameter filter clause for specific columns"""
        if not parameters:
            return "TRUE"
        
        # Build OR conditions for each parameter column
        conditions = []
        for param in parameters:
            column_info = self._get_column_mapping(param)
            if column_info:
                conditions.append(f"{column_info['column']} IS NOT NULL")
        
        if conditions:
            return "(" + " OR ".join(conditions) + ")"
        else:
            return "TRUE"
    
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
                
                # Log to query-specific logger if available
                if self.current_query_logger:
                    self.current_query_logger.info(f"✅ Query execution completed successfully")
                    self.current_query_logger.info(f"📊 Rows returned: {len(data)}")
                    self.current_query_logger.info(f"⏱️  Execution time: {execution_time:.3f} seconds")
                    if len(data) > 0:
                        self.current_query_logger.info(f"📋 Sample of first row: {dict(data.iloc[0]) if len(data) > 0 else 'No data'}")
                
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
                if self.current_query_logger:
                    self.current_query_logger.error(f"❌ Query execution failed: {str(e)}")
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