"""
Visualization Agent
Generates maps, charts, and plots for oceanographic data visualization
Uses plotly for interactive charts and folium for interactive maps
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
import base64
from io import BytesIO

from .base_agent import BaseAgent, AgentResult, LLMMessage

class VisualizationType(Enum):
    """Types of visualizations"""
    MAP = "map"                           # Geographic map
    TIME_SERIES = "time_series"           # Time series plot
    DEPTH_PROFILE = "depth_profile"       # Vertical profile
    SCATTER = "scatter"                   # Scatter plot
    HEATMAP = "heatmap"                   # Heat map
    HISTOGRAM = "histogram"               # Distribution histogram
    BOX_PLOT = "box_plot"                # Box plot
    CORRELATION_MATRIX = "correlation_matrix"  # Correlation heatmap
    COMPARISON = "comparison"             # Comparison charts
    SUMMARY_DASHBOARD = "summary_dashboard"    # Multi-panel dashboard

@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    viz_type: VisualizationType
    title: str
    parameters: List[str]
    color_scheme: str = "viridis"
    width: int = 800
    height: int = 600
    interactive: bool = True
    show_legend: bool = True

@dataclass
class VisualizationOutput:
    """Output from visualization generation"""
    viz_type: VisualizationType
    html_content: str
    json_config: Dict[str, Any]
    metadata: Dict[str, Any]
    file_size: int

class VisualizationAgent(BaseAgent):
    """Agent responsible for creating oceanographic data visualizations"""
    
    def __init__(self, config):
        super().__init__(config, "visualization")
        
        # Color schemes for different parameters
        self.color_schemes = {
            'TEMP': 'RdYlBu_r',       # Red-Yellow-Blue reversed for temperature
            'PSAL': 'viridis',        # Viridis for salinity
            'PRES': 'plasma',         # Plasma for pressure/depth
            'default': 'viridis'
        }
        
        # Parameter units and ranges
        self.parameter_info = {
            'TEMP': {'units': '°C', 'range': [-2, 30], 'name': 'Temperature'},
            'PSAL': {'units': 'PSU', 'range': [32, 38], 'name': 'Salinity'},
            'PRES': {'units': 'dbar', 'range': [0, 2000], 'name': 'Pressure'},
            'depth': {'units': 'm', 'range': [0, 2000], 'name': 'Depth'}
        }
        
        # Map settings
        self.map_settings = {
            'default_zoom': 4,
            'tiles': 'OpenStreetMap',
            'ocean_color': '#1f77b4',
            'marker_size': 5
        }
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Process data and generate visualizations"""
        
        try:
            if not isinstance(input_data, dict):
                return AgentResult.error_result(
                    self.agent_name,
                    ["Input must be a dictionary containing data and analysis results"]
                )
            
            # Extract data and analysis results
            data_records = input_data.get('data', [])
            analysis_results = input_data.get('analysis_results', {})
            metadata = input_data.get('metadata', {})
            
            if not data_records:
                return AgentResult.error_result(
                    self.agent_name,
                    ["No data provided for visualization"]
                )
            
            df = pd.DataFrame(data_records)
            
            # Determine visualization type from context or operator graph
            viz_configs = await self._determine_visualizations(df, analysis_results, context or {})
            
            self.logger.info(f"Generating {len(viz_configs)} visualizations")
            
            # Generate visualizations
            visualizations = {}
            for config in viz_configs:
                viz_output = await self._create_visualization(df, config, analysis_results)
                if viz_output:
                    visualizations[config.viz_type.value] = viz_output
            
            # Create summary dashboard if multiple visualizations
            if len(visualizations) > 1:
                dashboard = await self._create_dashboard(visualizations, df)
                if dashboard:
                    visualizations['dashboard'] = dashboard
            
            return AgentResult.success_result(
                self.agent_name,
                {
                    'visualizations': {k: {
                        'html': v.html_content,
                        'config': v.json_config,
                        'metadata': v.metadata,
                        'type': v.viz_type.value
                    } for k, v in visualizations.items()},
                    'summary': {
                        'total_visualizations': len(visualizations),
                        'data_points': len(df),
                        'parameters': list(df.get('parameter', df.select_dtypes(include=[np.number]).columns).unique())
                    }
                },
                {'visualizations_created': len(visualizations)}
            )
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to create visualizations: {str(e)}"]
            )
    
    async def _determine_visualizations(self, df: pd.DataFrame, analysis_results: Dict[str, Any], 
                                      context: Dict[str, Any]) -> List[VisualizationConfig]:
        """Determine what visualizations to create based on data and context"""
        
        configs = []
        
        # Check operator graph for visualization hints
        operator_graph = context.get('operator_graph', {})
        
        # Handle OperatorGraph object vs dictionary
        if hasattr(operator_graph, 'nodes'):
            nodes = operator_graph.nodes
        elif isinstance(operator_graph, dict):
            nodes = operator_graph.get('nodes', [])
        else:
            nodes = []
        
        viz_node = None
        for node in nodes:
            # Handle both dictionary and object nodes
            if hasattr(node, 'operation'):
                operation = node.operation
                parameters = getattr(node, 'parameters', {})
            elif isinstance(node, dict):
                operation = node.get('operation')
                parameters = node.get('parameters', {})
            else:
                continue
                
            if operation == 'visualization':
                viz_node = node
                break
        
        # Get requested visualization type
        if viz_node:
            if hasattr(viz_node, 'parameters'):
                requested_type = viz_node.parameters.get('type', 'summary_map')
            elif isinstance(viz_node, dict):
                requested_type = viz_node.get('parameters', {}).get('type', 'summary_map')
            else:
                requested_type = 'summary_map'
        else:
            requested_type = 'summary_map'
        
        # Determine available data columns
        has_spatial = all(col in df.columns for col in ['latitude', 'longitude'])
        has_temporal = 'profile_date' in df.columns
        has_depth = any(col in df.columns for col in ['pressure', 'depth'])
        has_parameters = 'parameter' in df.columns and 'parameter_value' in df.columns
        
        # Create configurations based on requested type and available data
        if requested_type == 'map_comparison' or requested_type == 'summary_map':
            if has_spatial:
                configs.append(VisualizationConfig(
                    viz_type=VisualizationType.MAP,
                    title="Oceanographic Data Locations",
                    parameters=list(df['parameter'].unique()) if has_parameters else ['value'],
                    color_scheme=self._get_color_scheme(df)
                ))
        
        if requested_type == 'time_series' or (has_temporal and len(df) > 10):
            configs.append(VisualizationConfig(
                viz_type=VisualizationType.TIME_SERIES,
                title="Time Series Analysis",
                parameters=list(df['parameter'].unique()) if has_parameters else ['value'],
                color_scheme=self._get_color_scheme(df)
            ))
        
        if requested_type == 'depth_profile' or (has_depth and has_parameters):
            configs.append(VisualizationConfig(
                viz_type=VisualizationType.DEPTH_PROFILE,
                title="Depth Profile",
                parameters=list(df['parameter'].unique()) if has_parameters else ['value'],
                color_scheme=self._get_color_scheme(df)
            ))
        
        if requested_type == 'parameter_comparison' and has_parameters:
            configs.append(VisualizationConfig(
                viz_type=VisualizationType.COMPARISON,
                title="Parameter Comparison",
                parameters=list(df['parameter'].unique()),
                color_scheme='Set2'
            ))
        
        # Add analysis-specific visualizations
        if 'correlation' in analysis_results:
            configs.append(VisualizationConfig(
                viz_type=VisualizationType.CORRELATION_MATRIX,
                title="Parameter Correlations",
                parameters=list(df['parameter'].unique()) if has_parameters else [],
                color_scheme='RdBu_r'
            ))
        
        if 'descriptive' in analysis_results:
            configs.append(VisualizationConfig(
                viz_type=VisualizationType.HISTOGRAM,
                title="Data Distribution",
                parameters=list(df['parameter'].unique()) if has_parameters else ['value'],
                color_scheme='viridis'
            ))
        
        # Default to map if no specific visualizations determined
        if not configs and has_spatial:
            configs.append(VisualizationConfig(
                viz_type=VisualizationType.MAP,
                title="Data Overview",
                parameters=['all'],
                color_scheme='viridis'
            ))
        
        return configs
    
    def _get_color_scheme(self, df: pd.DataFrame) -> str:
        """Get appropriate color scheme based on data"""
        if 'parameter' in df.columns:
            main_param = df['parameter'].value_counts().index[0]
            return self.color_schemes.get(main_param, self.color_schemes['default'])
        return self.color_schemes['default']
    
    async def _create_visualization(self, df: pd.DataFrame, config: VisualizationConfig, 
                                  analysis_results: Dict[str, Any]) -> Optional[VisualizationOutput]:
        """Create a single visualization"""
        
        try:
            if config.viz_type == VisualizationType.MAP:
                html_content = self._create_map(df, config)
            elif config.viz_type == VisualizationType.TIME_SERIES:
                html_content = self._create_time_series(df, config)
            elif config.viz_type == VisualizationType.DEPTH_PROFILE:
                html_content = self._create_depth_profile(df, config)
            elif config.viz_type == VisualizationType.CORRELATION_MATRIX:
                html_content = self._create_correlation_matrix(analysis_results.get('correlation', {}), config)
            elif config.viz_type == VisualizationType.HISTOGRAM:
                html_content = self._create_histogram(df, config)
            elif config.viz_type == VisualizationType.COMPARISON:
                html_content = self._create_comparison(df, config)
            else:
                html_content = self._create_summary_chart(df, config)
            
            if html_content:
                return VisualizationOutput(
                    viz_type=config.viz_type,
                    html_content=html_content,
                    json_config=self._config_to_dict(config),
                    metadata={'title': config.title, 'parameters': config.parameters},
                    file_size=len(html_content.encode('utf-8'))
                )
            
        except Exception as e:
            self.logger.error(f"Failed to create {config.viz_type.value} visualization: {e}")
            
        return None
    
    def _create_map(self, df: pd.DataFrame, config: VisualizationConfig) -> str:
        """Create interactive map visualization"""
        
        if not all(col in df.columns for col in ['latitude', 'longitude']):
            return self._create_error_html("Missing latitude/longitude data for map")
        
        # Calculate map center
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.map_settings['default_zoom'],
            tiles=self.map_settings['tiles']
        )
        
        # Add data points
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            # Color by parameter value
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param]
                
                if len(param_data) > 0:
                    # Create color map
                    values = param_data['parameter_value']
                    colormap = self._create_colormap(values, config.color_scheme)
                    
                    # Add markers
                    for _, row in param_data.iterrows():
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=self.map_settings['marker_size'],
                            color=colormap(row['parameter_value']),
                            fillColor=colormap(row['parameter_value']),
                            fillOpacity=0.7,
                            popup=f"{param}: {row['parameter_value']:.2f}\\nLat: {row['latitude']:.3f}\\nLon: {row['longitude']:.3f}"
                        ).add_to(m)
        else:
            # Simple markers
            for _, row in df.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=self.map_settings['marker_size'],
                    color=self.map_settings['ocean_color'],
                    fillColor=self.map_settings['ocean_color'],
                    fillOpacity=0.7,
                    popup=f"Lat: {row['latitude']:.3f}\\nLon: {row['longitude']:.3f}"
                ).add_to(m)
        
        # Add layer control if multiple parameters
        if 'parameter' in df.columns and df['parameter'].nunique() > 1:
            folium.LayerControl().add_to(m)
        
        return m._repr_html_()
    
    def _create_time_series(self, df: pd.DataFrame, config: VisualizationConfig) -> str:
        """Create time series visualization"""
        
        if 'profile_date' not in df.columns:
            return self._create_error_html("Missing date data for time series")
        
        df = df.copy()
        df['profile_date'] = pd.to_datetime(df['profile_date'])
        
        fig = go.Figure()
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param].sort_values('profile_date')
                
                param_info = self.parameter_info.get(param, {'name': param, 'units': ''})
                
                fig.add_trace(go.Scatter(
                    x=param_data['profile_date'],
                    y=param_data['parameter_value'],
                    mode='lines+markers',
                    name=f"{param_info['name']} ({param_info['units']})",
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        else:
            # Plot numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                fig.add_trace(go.Scatter(
                    x=df['profile_date'],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title='Date',
            yaxis_title='Value',
            width=config.width,
            height=config.height,
            hovermode='x unified',
            showlegend=config.show_legend
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id=f"timeseries_{hash(config.title)}")
    
    def _create_depth_profile(self, df: pd.DataFrame, config: VisualizationConfig) -> str:
        """Create depth profile visualization"""
        
        depth_col = 'pressure' if 'pressure' in df.columns else 'depth'
        if depth_col not in df.columns:
            return self._create_error_html(f"Missing {depth_col} data for depth profile")
        
        if 'parameter_value' not in df.columns or 'parameter' not in df.columns:
            return self._create_error_html("Missing parameter data for depth profile")
        
        fig = go.Figure()
        
        for param in df['parameter'].unique():
            param_data = df[df['parameter'] == param].sort_values(depth_col)
            
            param_info = self.parameter_info.get(param, {'name': param, 'units': ''})
            
            fig.add_trace(go.Scatter(
                x=param_data['parameter_value'],
                y=param_data[depth_col],
                mode='lines+markers',
                name=f"{param_info['name']} ({param_info['units']})",
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title='Value',
            yaxis_title='Depth (m)' if depth_col == 'depth' else 'Pressure (dbar)',
            yaxis=dict(autorange='reversed'),  # Depth increases downward
            width=config.width,
            height=config.height,
            hovermode='y unified',
            showlegend=config.show_legend
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id=f"depth_{hash(config.title)}")
    
    def _create_correlation_matrix(self, correlation_data: Dict[str, Any], config: VisualizationConfig) -> str:
        """Create correlation matrix heatmap"""
        
        if 'correlation_matrix' not in correlation_data:
            return self._create_error_html("No correlation matrix data available")
        
        corr_matrix = correlation_data['correlation_matrix']
        
        # Convert to DataFrame if it's a dict
        if isinstance(corr_matrix, dict):
            corr_df = pd.DataFrame(corr_matrix)
        else:
            corr_df = corr_matrix
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=list(corr_df.columns),
            y=list(corr_df.index),
            colorscale=config.color_scheme,
            zmid=0,
            text=corr_df.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            xaxis=dict(side='bottom'),
            yaxis=dict(side='left')
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id=f"correlation_{hash(config.title)}")
    
    def _create_histogram(self, df: pd.DataFrame, config: VisualizationConfig) -> str:
        """Create histogram/distribution plot"""
        
        fig = go.Figure()
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param]['parameter_value'].dropna()
                
                param_info = self.parameter_info.get(param, {'name': param, 'units': ''})
                
                fig.add_trace(go.Histogram(
                    x=param_data,
                    name=f"{param_info['name']}",
                    opacity=0.7,
                    nbinsx=30
                ))
        else:
            # Plot numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                fig.add_trace(go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    opacity=0.7,
                    nbinsx=30
                ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title='Value',
            yaxis_title='Frequency',
            width=config.width,
            height=config.height,
            barmode='overlay',
            showlegend=config.show_legend
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id=f"histogram_{hash(config.title)}")
    
    def _create_comparison(self, df: pd.DataFrame, config: VisualizationConfig) -> str:
        """Create comparison visualization (box plot)"""
        
        if 'parameter_value' not in df.columns or 'parameter' not in df.columns:
            return self._create_error_html("Missing parameter data for comparison")
        
        fig = go.Figure()
        
        for param in df['parameter'].unique():
            param_data = df[df['parameter'] == param]['parameter_value'].dropna()
            
            param_info = self.parameter_info.get(param, {'name': param, 'units': ''})
            
            fig.add_trace(go.Box(
                y=param_data,
                name=f"{param_info['name']}",
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title='Parameter',
            yaxis_title='Value',
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id=f"comparison_{hash(config.title)}")
    
    def _create_summary_chart(self, df: pd.DataFrame, config: VisualizationConfig) -> str:
        """Create summary chart when no specific type is determined"""
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            return self._create_comparison(df, config)
        else:
            return self._create_histogram(df, config)
    
    def _create_colormap(self, values: pd.Series, scheme: str):
        """Create a color mapping function for values"""
        
        def colormap(value):
            # Simple implementation - normalize value to 0-1 and map to color
            if len(values) == 0:
                return '#1f77b4'
            
            normalized = (value - values.min()) / (values.max() - values.min()) if values.max() != values.min() else 0.5
            
            # Simple color interpolation (blue to red)
            if scheme == 'RdYlBu_r':
                r = int(255 * normalized)
                b = int(255 * (1 - normalized))
                return f'rgb({r}, 100, {b})'
            else:
                # Default to blue gradient
                intensity = int(255 * (1 - normalized * 0.7))
                return f'rgb(30, {intensity}, 180)'
        
        return colormap
    
    async def _create_dashboard(self, visualizations: Dict[str, VisualizationOutput], 
                              df: pd.DataFrame) -> Optional[VisualizationOutput]:
        """Create a summary dashboard with multiple visualizations"""
        
        try:
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Oceanographic Data Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                    .viz-panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; }}
                    .summary {{ grid-column: 1 / -1; background: #f8f9fa; padding: 20px; margin-bottom: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>Oceanographic Data Analysis Dashboard</h1>
                
                <div class="summary">
                    <h2>Data Summary</h2>
                    <p><strong>Total Records:</strong> {len(df)}</p>
                    <p><strong>Parameters:</strong> {', '.join(df['parameter'].unique()) if 'parameter' in df.columns else 'Multiple'}</p>
                    <p><strong>Date Range:</strong> {self._get_date_range_string(df)}</p>
                    <p><strong>Visualizations:</strong> {len(visualizations)}</p>
                </div>
                
                <div class="dashboard">
            """
            
            for viz_name, viz_output in visualizations.items():
                dashboard_html += f"""
                    <div class="viz-panel">
                        <h3>{viz_output.metadata.get('title', viz_name.title())}</h3>
                        {viz_output.html_content}
                    </div>
                """
            
            dashboard_html += """
                </div>
            </body>
            </html>
            """
            
            return VisualizationOutput(
                viz_type=VisualizationType.SUMMARY_DASHBOARD,
                html_content=dashboard_html,
                json_config={'type': 'dashboard', 'panels': list(visualizations.keys())},
                metadata={'title': 'Data Dashboard', 'panel_count': len(visualizations)},
                file_size=len(dashboard_html.encode('utf-8'))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return None
    
    def _create_error_html(self, error_message: str) -> str:
        """Create HTML for error message"""
        return f"""
        <div style="padding: 20px; border: 1px solid #ff6b6b; background-color: #ffe0e0; border-radius: 5px;">
            <h3 style="color: #d63031; margin: 0;">Visualization Error</h3>
            <p style="margin: 10px 0 0 0;">{error_message}</p>
        </div>
        """
    
    def _get_date_range_string(self, df: pd.DataFrame) -> str:
        """Get formatted date range string"""
        if 'profile_date' in df.columns:
            dates = pd.to_datetime(df['profile_date'])
            return f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
        return "Date range not available"
    
    def _config_to_dict(self, config: VisualizationConfig) -> Dict[str, Any]:
        """Convert VisualizationConfig to dictionary"""
        return {
            'type': config.viz_type.value,
            'title': config.title,
            'parameters': config.parameters,
            'color_scheme': config.color_scheme,
            'width': config.width,
            'height': config.height,
            'interactive': config.interactive,
            'show_legend': config.show_legend
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about visualization capabilities"""
        return {
            "description": "Creates interactive maps, charts, and plots for oceanographic data",
            "visualization_types": [vtype.value for vtype in VisualizationType],
            "supported_parameters": list(self.parameter_info.keys()),
            "libraries": ["plotly", "folium"],
            "output_formats": ["HTML", "interactive"]
        }