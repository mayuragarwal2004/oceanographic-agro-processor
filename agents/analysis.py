"""
Analysis Agent
Performs statistical computations, trend analysis, and anomaly detection
on retrieved oceanographic data
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from .base_agent import BaseAgent, AgentResult, LLMMessage

class AnalysisType(Enum):
    """Types of analysis operations"""
    DESCRIPTIVE = "descriptive"           # Mean, median, std, etc.
    TREND = "trend"                       # Temporal trend analysis
    ANOMALY = "anomaly"                   # Outlier/anomaly detection  
    CORRELATION = "correlation"           # Parameter correlations
    SEASONAL = "seasonal"                 # Seasonal analysis
    SPATIAL = "spatial"                   # Spatial patterns
    COMPARATIVE = "comparative"           # Compare groups/regions
    CLIMATOLOGY = "climatology"           # Long-term climate analysis

@dataclass
class AnalysisMetrics:
    """Statistical metrics for oceanographic data"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    count: int
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

@dataclass
class TrendResult:
    """Results from trend analysis"""
    parameter: str
    trend_slope: float
    trend_intercept: float
    p_value: float
    r_squared: float
    confidence_interval: Tuple[float, float]
    trend_direction: str  # increasing, decreasing, stable
    significance: str     # significant, not_significant

@dataclass
class AnomalyResult:
    """Results from anomaly detection"""
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    threshold: float
    method: str
    anomaly_count: int
    anomaly_percentage: float

@dataclass 
class SeasonalResult:
    """Results from seasonal analysis"""
    parameter: str
    seasonal_means: Dict[str, float]
    seasonal_stds: Dict[str, float]
    seasonal_peaks: Dict[str, Tuple[str, float]]
    amplitude: float
    phase_shift: float

class AnalysisAgent(BaseAgent):
    """Agent responsible for statistical analysis of oceanographic data"""
    
    def __init__(self, config):
        super().__init__(config, "analysis")
        
        # Analysis parameters
        self.anomaly_threshold = 2.5  # Standard deviations for anomaly detection
        self.trend_significance = 0.05  # P-value threshold for trend significance
        self.seasonal_months = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5], 
            'summer': [6, 7, 8],
            'fall': [9, 10, 11]
        }
        
        # Quality control thresholds (typical oceanographic ranges)
        self.quality_thresholds = {
            'TEMP': {'min': -2.5, 'max': 35.0, 'units': '°C'},
            'PSAL': {'min': 2.0, 'max': 42.0, 'units': 'PSU'},
            'PRES': {'min': 0.0, 'max': 6000.0, 'units': 'dbar'}
        }
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Process data and perform requested analysis"""
        
        try:
            if not isinstance(input_data, dict) or 'data' not in input_data:
                return AgentResult.error_result(
                    self.agent_name,
                    ["Input must be a dictionary with 'data' key containing DataFrame records"]
                )
            
            # Convert data records back to DataFrame
            data_records = input_data['data']
            if not data_records:
                return AgentResult.error_result(
                    self.agent_name,
                    ["No data records provided for analysis"]
                )
            
            df = pd.DataFrame(data_records)
            
            # Get analysis requirements from context
            analysis_types = self._determine_analysis_types(context or {})
            
            self.logger.info(f"Performing {len(analysis_types)} types of analysis on {len(df)} records")
            
            # Perform data quality checks
            quality_report = self._check_data_quality(df)
            
            # Perform requested analyses
            analysis_results = {}
            
            for analysis_type in analysis_types:
                if analysis_type == AnalysisType.DESCRIPTIVE:
                    analysis_results['descriptive'] = await self._descriptive_analysis(df)
                elif analysis_type == AnalysisType.TREND:
                    analysis_results['trend'] = await self._trend_analysis(df)
                elif analysis_type == AnalysisType.ANOMALY:
                    analysis_results['anomaly'] = await self._anomaly_detection(df)
                elif analysis_type == AnalysisType.CORRELATION:
                    analysis_results['correlation'] = await self._correlation_analysis(df)
                elif analysis_type == AnalysisType.SEASONAL:
                    analysis_results['seasonal'] = await self._seasonal_analysis(df)
                elif analysis_type == AnalysisType.SPATIAL:
                    analysis_results['spatial'] = await self._spatial_analysis(df)
                elif analysis_type == AnalysisType.COMPARATIVE:
                    analysis_results['comparative'] = await self._comparative_analysis(df, context)
                elif analysis_type == AnalysisType.CLIMATOLOGY:
                    analysis_results['climatology'] = await self._climatology_analysis(df)
            
            # Generate insights using LLM
            insights = await self._generate_insights(analysis_results, df)
            
            return AgentResult.success_result(
                self.agent_name,
                {
                    'analysis_results': analysis_results,
                    'quality_report': quality_report,
                    'insights': insights,
                    'data_summary': {
                        'total_records': len(df),
                        'parameters': list(df.get('parameter', df.columns).unique()) if 'parameter' in df.columns else list(df.columns),
                        'date_range': self._get_date_range(df)
                    }
                },
                {'analyses_performed': len(analysis_results)}
            )
            
        except Exception as e:
            self.logger.error(f"Error performing analysis: {str(e)}")
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to perform analysis: {str(e)}"]
            )
    
    def _determine_analysis_types(self, context: Dict[str, Any]) -> List[AnalysisType]:
        """Determine which analyses to perform based on context"""
        
        analyses = [AnalysisType.DESCRIPTIVE]  # Always include descriptive stats
        
        # Check for specific requests in operator graph
        operator_graph = context.get('operator_graph', {})
        nodes = operator_graph.get('nodes', [])
        
        for node in nodes:
            if node.get('operation') == 'statistical_analysis':
                operations = node.get('parameters', {}).get('operations', [])
                
                if any('trend' in op for op in operations):
                    analyses.append(AnalysisType.TREND)
                if any('anomaly' in op for op in operations):
                    analyses.append(AnalysisType.ANOMALY)
                if any('correlation' in op for op in operations):
                    analyses.append(AnalysisType.CORRELATION)
                if any('seasonal' in op for op in operations):
                    analyses.append(AnalysisType.SEASONAL)
                    
            elif node.get('operation') == 'comparison':
                comp_type = node.get('parameters', {}).get('type', '')
                if comp_type == 'spatial':
                    analyses.append(AnalysisType.COMPARATIVE)
                elif comp_type == 'temporal':
                    analyses.append(AnalysisType.TREND)
        
        return list(set(analyses))  # Remove duplicates
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality checks"""
        
        quality_report = {
            'total_records': len(df),
            'missing_values': {},
            'out_of_range': {},
            'quality_score': 1.0,
            'issues': []
        }
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                quality_report['missing_values'][col] = missing_count
                quality_report['issues'].append(f"{missing_count} missing values in {col}")
        
        # Check parameter value ranges
        if 'parameter' in df.columns and 'parameter_value' in df.columns:
            for param in df['parameter'].unique():
                if param in self.quality_thresholds:
                    param_data = df[df['parameter'] == param]['parameter_value']
                    threshold = self.quality_thresholds[param]
                    
                    out_of_range = param_data[(param_data < threshold['min']) | 
                                             (param_data > threshold['max'])]
                    
                    if len(out_of_range) > 0:
                        quality_report['out_of_range'][param] = len(out_of_range)
                        quality_report['issues'].append(
                            f"{len(out_of_range)} {param} values outside expected range "
                            f"({threshold['min']}-{threshold['max']} {threshold['units']})"
                        )
        
        # Calculate overall quality score
        total_issues = (sum(quality_report['missing_values'].values()) + 
                       sum(quality_report['out_of_range'].values()))
        
        if quality_report['total_records'] > 0:
            quality_report['quality_score'] = max(0.0, 1.0 - (total_issues / quality_report['total_records']))
        
        return quality_report
    
    async def _descriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        
        results = {}
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            # Analysis by parameter
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param]['parameter_value']
                
                if len(param_data) > 0:
                    results[param] = self._calculate_metrics(param_data).to_dict()
        else:
            # Analysis of numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                results[col] = self._calculate_metrics(df[col]).to_dict()
        
        return results
    
    def _calculate_metrics(self, data: pd.Series) -> AnalysisMetrics:
        """Calculate statistical metrics for a data series"""
        
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return AnalysisMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        return AnalysisMetrics(
            mean=float(clean_data.mean()),
            median=float(clean_data.median()),
            std=float(clean_data.std()),
            min=float(clean_data.min()),
            max=float(clean_data.max()),
            q25=float(clean_data.quantile(0.25)),
            q75=float(clean_data.quantile(0.75)),
            count=len(clean_data),
            skewness=float(clean_data.skew()) if len(clean_data) > 2 else None,
            kurtosis=float(clean_data.kurtosis()) if len(clean_data) > 3 else None
        )
    
    async def _trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform trend analysis on time series data"""
        
        results = {}
        
        if 'profile_date' not in df.columns:
            return {'error': 'No date column found for trend analysis'}
        
        # Convert dates to numeric for regression
        df = df.copy()
        df['profile_date'] = pd.to_datetime(df['profile_date'])
        df['date_numeric'] = (df['profile_date'] - df['profile_date'].min()).dt.days
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param]
                
                if len(param_data) > 5:  # Need sufficient data for trend analysis
                    trend_result = self._calculate_trend(
                        param_data['date_numeric'], 
                        param_data['parameter_value']
                    )
                    results[param] = trend_result.to_dict() if trend_result else None
        
        return results
    
    def _calculate_trend(self, x: pd.Series, y: pd.Series) -> Optional[TrendResult]:
        """Calculate trend statistics using linear regression"""
        
        # Remove NaN values
        valid_mask = ~(pd.isna(x) | pd.isna(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < 3:
            return None
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            # Calculate confidence interval for slope
            t_val = stats.t.ppf(0.975, len(x_clean) - 2)  # 95% confidence
            ci_lower = slope - t_val * std_err
            ci_upper = slope + t_val * std_err
            
            # Determine trend direction
            if abs(slope) < std_err:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            # Determine significance
            significance = "significant" if p_value < self.trend_significance else "not_significant"
            
            return TrendResult(
                parameter=str(y.name) if hasattr(y, 'name') else 'value',
                trend_slope=slope,
                trend_intercept=intercept,
                p_value=p_value,
                r_squared=r_value**2,
                confidence_interval=(ci_lower, ci_upper),
                trend_direction=trend_direction,
                significance=significance
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate trend: {e}")
            return None
    
    async def _anomaly_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        
        results = {}
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param]['parameter_value'].dropna()
                
                if len(param_data) > 10:  # Need sufficient data
                    anomaly_result = self._detect_anomalies_zscore(param_data)
                    results[param] = anomaly_result.to_dict() if anomaly_result else None
        else:
            # Analyze numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                anomaly_result = self._detect_anomalies_zscore(df[col].dropna())
                results[col] = anomaly_result.to_dict() if anomaly_result else None
        
        return results
    
    def _detect_anomalies_zscore(self, data: pd.Series) -> Optional[AnomalyResult]:
        """Detect anomalies using Z-score method"""
        
        if len(data) < 10:
            return None
        
        try:
            z_scores = np.abs(stats.zscore(data))
            anomaly_mask = z_scores > self.anomaly_threshold
            
            anomaly_indices = data.index[anomaly_mask].tolist()
            anomaly_scores = z_scores[anomaly_mask].tolist()
            
            return AnomalyResult(
                anomaly_indices=anomaly_indices,
                anomaly_scores=anomaly_scores,
                threshold=self.anomaly_threshold,
                method="z_score",
                anomaly_count=len(anomaly_indices),
                anomaly_percentage=len(anomaly_indices) / len(data) * 100
            )
        except Exception as e:
            self.logger.warning(f"Failed to detect anomalies: {e}")
            return None
    
    async def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between parameters"""
        
        if 'parameter' not in df.columns or 'parameter_value' not in df.columns:
            return {'error': 'Cannot perform correlation analysis without parameter columns'}
        
        # Pivot data to get parameters as columns
        pivot_df = df.pivot_table(
            index=['profile_date', 'latitude', 'longitude'], 
            columns='parameter', 
            values='parameter_value',
            aggfunc='mean'
        ).reset_index()
        
        # Calculate correlation matrix
        numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = pivot_df[numeric_cols].corr()
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': self._find_strong_correlations(correlation_matrix)
            }
        
        return {'error': 'Insufficient numeric data for correlation analysis'}
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find pairs of variables with strong correlations"""
        
        strong_correls = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= threshold:
                    strong_correls.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'strong positive' if corr_val > 0 else 'strong negative'
                    })
        
        return strong_correls
    
    async def _seasonal_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in the data"""
        
        if 'profile_date' not in df.columns:
            return {'error': 'No date column found for seasonal analysis'}
        
        df = df.copy()
        df['profile_date'] = pd.to_datetime(df['profile_date'])
        df['month'] = df['profile_date'].dt.month
        
        # Map months to seasons
        def get_season(month):
            if month in self.seasonal_months['winter']:
                return 'winter'
            elif month in self.seasonal_months['spring']:
                return 'spring'
            elif month in self.seasonal_months['summer']:
                return 'summer'
            else:
                return 'fall'
        
        df['season'] = df['month'].apply(get_season)
        
        results = {}
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param]
                
                seasonal_stats = param_data.groupby('season')['parameter_value'].agg(['mean', 'std']).to_dict()
                
                # Find seasonal peaks
                seasonal_means = seasonal_stats['mean']
                peak_season = max(seasonal_means, key=seasonal_means.get)
                
                results[param] = {
                    'seasonal_means': seasonal_means,
                    'seasonal_stds': seasonal_stats['std'],
                    'peak_season': peak_season,
                    'peak_value': seasonal_means[peak_season],
                    'amplitude': max(seasonal_means.values()) - min(seasonal_means.values())
                }
        
        return results
    
    async def _spatial_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spatial patterns in the data"""
        
        if not all(col in df.columns for col in ['latitude', 'longitude']):
            return {'error': 'Missing latitude/longitude columns for spatial analysis'}
        
        results = {
            'spatial_extent': {
                'lat_range': [df['latitude'].min(), df['latitude'].max()],
                'lon_range': [df['longitude'].min(), df['longitude'].max()],
                'center_lat': df['latitude'].mean(),
                'center_lon': df['longitude'].mean()
            },
            'spatial_distribution': {}
        }
        
        # Create spatial bins for analysis
        lat_bins = pd.cut(df['latitude'], bins=5, labels=['S', 'S-C', 'C', 'C-N', 'N'])
        lon_bins = pd.cut(df['longitude'], bins=5, labels=['W', 'W-C', 'C', 'C-E', 'E'])
        
        df = df.copy()
        df['lat_bin'] = lat_bins
        df['lon_bin'] = lon_bins
        df['spatial_bin'] = df['lat_bin'].astype(str) + '-' + df['lon_bin'].astype(str)
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param]
                
                spatial_stats = param_data.groupby('spatial_bin')['parameter_value'].agg(['mean', 'std', 'count'])
                results['spatial_distribution'][param] = spatial_stats.to_dict('index')
        
        return results
    
    async def _comparative_analysis(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis between groups"""
        
        # This would compare different locations or time periods
        # Implementation depends on the specific comparison requested
        
        results = {
            'comparison_type': 'location',  # Default
            'groups': [],
            'statistics': {},
            'significance_tests': {}
        }
        
        # Simplified implementation - could be expanded
        if 'location_group' in df.columns:
            groups = df['location_group'].unique()
            results['groups'] = list(groups)
            
            if 'parameter_value' in df.columns and 'parameter' in df.columns:
                for param in df['parameter'].unique():
                    param_data = df[df['parameter'] == param]
                    
                    group_stats = {}
                    for group in groups:
                        group_data = param_data[param_data['location_group'] == group]['parameter_value']
                        group_stats[group] = self._calculate_metrics(group_data).to_dict()
                    
                    results['statistics'][param] = group_stats
        
        return results
    
    async def _climatology_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform climatological analysis (long-term patterns)"""
        
        if 'profile_date' not in df.columns:
            return {'error': 'No date column found for climatology analysis'}
        
        df = df.copy()
        df['profile_date'] = pd.to_datetime(df['profile_date'])
        df['year'] = df['profile_date'].dt.year
        df['month'] = df['profile_date'].dt.month
        
        results = {
            'time_range': {
                'start_year': int(df['year'].min()),
                'end_year': int(df['year'].max()),
                'total_years': int(df['year'].max() - df['year'].min() + 1)
            },
            'monthly_climatology': {},
            'annual_means': {}
        }
        
        if 'parameter_value' in df.columns and 'parameter' in df.columns:
            for param in df['parameter'].unique():
                param_data = df[df['parameter'] == param]
                
                # Monthly climatology
                monthly_clim = param_data.groupby('month')['parameter_value'].agg(['mean', 'std']).to_dict()
                results['monthly_climatology'][param] = monthly_clim
                
                # Annual means
                annual_means = param_data.groupby('year')['parameter_value'].mean().to_dict()
                results['annual_means'][param] = annual_means
        
        return results
    
    async def _generate_insights(self, analysis_results: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate natural language insights from analysis results"""
        
        # Create a summary of key findings
        summary_data = {
            'data_summary': {
                'total_records': len(df),
                'parameters': list(df['parameter'].unique()) if 'parameter' in df.columns else 'multiple',
                'time_span': self._get_date_range(df)
            },
            'analysis_types': list(analysis_results.keys()),
            'key_findings': {}
        }
        
        # Extract key findings from each analysis
        for analysis_type, results in analysis_results.items():
            if analysis_type == 'descriptive' and isinstance(results, dict):
                for param, stats in results.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        summary_data['key_findings'][f'{param}_average'] = stats['mean']
            
            elif analysis_type == 'trend' and isinstance(results, dict):
                for param, trend in results.items():
                    if isinstance(trend, dict) and 'trend_direction' in trend:
                        summary_data['key_findings'][f'{param}_trend'] = trend['trend_direction']
        
        # Generate insights using LLM
        system_prompt = """You are an oceanographic data analyst. Given analysis results, generate 3-5 key insights in natural language.
        Focus on:
        1. Notable patterns or trends
        2. Unusual findings or anomalies  
        3. Relationships between parameters
        4. Seasonal or spatial patterns
        5. Data quality observations
        
        Keep insights scientific but accessible. Mention specific values when relevant."""
        
        user_prompt = f"Generate insights from this oceanographic analysis: {json.dumps(summary_data, default=str)}"
        
        try:
            messages = [
                self.create_system_message(system_prompt),
                self.create_user_message(user_prompt)
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            # Parse insights (assume they're returned as a list or numbered items)
            insights_text = response.content
            
            # Split by newlines and filter out empty lines
            insights = [line.strip() for line in insights_text.split('\n') 
                       if line.strip() and not line.strip().isdigit()]
            
            return insights[:5]  # Limit to 5 insights
            
        except Exception as e:
            self.logger.warning(f"Failed to generate insights with LLM: {e}")
            return ["Analysis completed successfully."]
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get date range information from DataFrame"""
        
        if 'profile_date' in df.columns:
            dates = pd.to_datetime(df['profile_date'])
            return {
                'start': dates.min().strftime('%Y-%m-%d'),
                'end': dates.max().strftime('%Y-%m-%d'),
                'span_days': (dates.max() - dates.min()).days
            }
        
        return {'start': None, 'end': None, 'span_days': 0}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about analysis capabilities"""
        return {
            "description": "Performs statistical analysis on oceanographic data",
            "analysis_types": [atype.value for atype in AnalysisType],
            "supported_parameters": list(self.quality_thresholds.keys()),
            "quality_checks": "Range validation, missing value detection",
            "statistical_methods": ["descriptive", "regression", "z-score anomaly detection", "correlation"]
        }