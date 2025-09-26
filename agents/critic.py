"""
Critic Agent
Validates results for scientific accuracy and checks for data quality issues
Provides quality assurance and scientific review of analysis results
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .base_agent import BaseAgent, AgentResult, LLMMessage
from utils.query_logging import QueryLogger

class ValidationLevel(Enum):
    """Levels of validation checks"""
    BASIC = "basic"           # Basic data validation
    SCIENTIFIC = "scientific" # Scientific plausibility checks  
    STATISTICAL = "statistical" # Statistical validity checks
    COMPREHENSIVE = "comprehensive" # All validation types

class IssueLevel(Enum):
    """Severity levels for identified issues"""
    INFO = "info"         # Informational notice
    WARNING = "warning"   # Potential issue
    ERROR = "error"       # Serious problem
    CRITICAL = "critical" # Critical data quality issue

@dataclass
class ValidationIssue:
    """Represents a validation issue found in the data or analysis"""
    issue_id: str
    level: IssueLevel
    category: str
    description: str
    affected_data: str
    recommendation: str
    metadata: Dict[str, Any] = None

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    overall_score: float  # 0-100 score
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    recommendations: List[str]
    approved: bool

class CriticAgent(BaseAgent):
    """Agent responsible for validating scientific accuracy and data quality"""
    
    def __init__(self, config):
        super().__init__(config, "critic")
        
        # Query logging
        self.query_logger_manager = None
        self.current_query_logger = None
        
        # Scientific parameter ranges (oceanographic standards)
        self.parameter_ranges = {
            'TEMP': {
                'global_min': -2.5, 'global_max': 35.0,  # Global ocean range
                'surface_max': 32.0,  # Typical surface maximum
                'deep_max': 4.0,      # Deep ocean maximum  
                'units': '°C'
            },
            'PSAL': {
                'global_min': 2.0, 'global_max': 42.0,   # Global salinity range
                'typical_min': 32.0, 'typical_max': 37.5, # Typical open ocean
                'units': 'PSU'
            },
            'PRES': {
                'global_min': 0.0, 'global_max': 11000.0, # Ocean depth range
                'abyssal_min': 4000.0,  # Abyssal depths
                'units': 'dbar'
            }
        }
        
        # Quality control flag meanings (NOAA standards)
        self.qc_flags = {
            '0': 'No QC performed',
            '1': 'Good data',
            '2': 'Probably good data',
            '3': 'Probably bad data',
            '4': 'Bad data',
            '5': 'Changed data',
            '8': 'Estimated data',
            '9': 'Missing data'
        }
        
        # Statistical thresholds
        self.stat_thresholds = {
            'outlier_iqr_factor': 3.0,    # IQR multiplier for outlier detection
            'correlation_threshold': 0.95, # Suspicious correlation threshold
            'trend_significance': 0.05,    # P-value for trend significance
            'min_sample_size': 10         # Minimum samples for statistical tests
        }
        
        # Validation rules
        self.validation_rules = {
            'temperature_depth': self._validate_temperature_depth_relationship,
            'salinity_range': self._validate_salinity_range,
            'pressure_consistency': self._validate_pressure_consistency,
            'temporal_consistency': self._validate_temporal_consistency,
            'spatial_consistency': self._validate_spatial_consistency,
            'statistical_validity': self._validate_statistical_results
        }
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Validate data and analysis results"""
        
        # Initialize query logger from shared context
        self.current_query_logger = context.get('current_query_logger') if context else None
        if not self.current_query_logger and self.query_logger_manager:
            # Fallback to creating new logger if not provided in context
            self.current_query_logger = self.query_logger_manager.get_query_logger_wrapper()
        
        if self.current_query_logger:
            self.current_query_logger.log_agent_start("critic", {
                "input_type": type(input_data).__name__,
                "data_length": len(input_data.get('data', []) if isinstance(input_data, dict) else []),
                "has_analysis_results": 'analysis_results' in input_data if isinstance(input_data, dict) else False,
                "has_visualizations": 'visualizations' in input_data if isinstance(input_data, dict) else False,
                "context_keys": list(context.keys()) if context else []
            })
        
        try:
            if not isinstance(input_data, dict):
                error_msg = "Input must be a dictionary containing data and analysis results"
                if self.current_query_logger:
                    self.current_query_logger.log_error("critic", error_msg)
                return AgentResult.error_result(
                    self.agent_name,
                    [error_msg]
                )
            
            # Extract components to validate
            raw_data = pd.DataFrame(input_data.get('data', []))
            analysis_results = input_data.get('analysis_results', {})
            visualizations = input_data.get('visualizations', {})
            metadata = input_data.get('metadata', {})
            
            # Determine validation level
            validation_level = self._determine_validation_level(context or {})
            
            if self.current_query_logger:
                self.current_query_logger.info(f"Performing {validation_level.value} validation on {len(raw_data)} records")
                self.current_query_logger.info(f"Components to validate: data={len(raw_data)}, analysis={len(analysis_results)}, visualizations={len(visualizations)}")
            
            self.logger.info(f"Performing {validation_level.value} validation on {len(raw_data)} records")
            
            # Perform validation checks
            validation_report = await self._perform_validation(
                raw_data, analysis_results, visualizations, validation_level
            )
            
            # Generate recommendations using LLM
            enhanced_recommendations = await self._generate_recommendations(validation_report)
            validation_report.recommendations.extend(enhanced_recommendations)
            
            result = {
                'validation_report': {
                    'overall_score': validation_report.overall_score,
                    'approved': validation_report.approved,
                    'issues': [self._issue_to_dict(issue) for issue in validation_report.issues],
                    'summary': validation_report.summary,
                    'recommendations': validation_report.recommendations
                },
                'quality_metrics': self._calculate_quality_metrics(raw_data, validation_report),
                'validation_metadata': {
                    'level': validation_level.value,
                    'timestamp': datetime.now().isoformat(),
                    'rules_applied': list(self.validation_rules.keys())
                }
            }
            
            
            if self.current_query_logger:
                self.current_query_logger.log_result("critic", {
                    'overall_score': validation_report.overall_score,
                    'approved': validation_report.approved,
                    'issues_found': len(validation_report.issues),
                    'validation_level': validation_level.value
                })
            
            return AgentResult.success_result(
                self.agent_name,
                result,
                {'issues_found': len(validation_report.issues)}
            )
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            
            if self.current_query_logger:
                self.current_query_logger.log_error("critic", str(e))
                
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to validate results: {str(e)}"]
            )
        finally:
            # Set logger to None (orchestrator handles cleanup for shared logger)
            self.current_query_logger = None
    
    def _determine_validation_level(self, context: Dict[str, Any]) -> ValidationLevel:
        """Determine appropriate validation level based on context"""
        
        # Check for specific validation requirements
        requested_level = context.get('validation_level', 'scientific')
        
        try:
            return ValidationLevel(requested_level)
        except ValueError:
            return ValidationLevel.SCIENTIFIC  # Default
    
    async def _perform_validation(self, data: pd.DataFrame, analysis_results: Dict[str, Any], 
                                 visualizations: Dict[str, Any], level: ValidationLevel) -> ValidationReport:
        """Perform comprehensive validation"""
        
        issues = []
        
        # Basic data validation (always performed)
        issues.extend(await self._validate_data_basic(data))
        
        # Scientific validation
        if level in [ValidationLevel.SCIENTIFIC, ValidationLevel.COMPREHENSIVE]:
            issues.extend(await self._validate_data_scientific(data))
        
        # Statistical validation
        if level in [ValidationLevel.STATISTICAL, ValidationLevel.COMPREHENSIVE]:
            issues.extend(await self._validate_analysis_statistical(analysis_results))
        
        # Comprehensive validation includes all checks plus cross-validation
        if level == ValidationLevel.COMPREHENSIVE:
            issues.extend(await self._validate_cross_consistency(data, analysis_results))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(issues, len(data))
        
        # Determine approval status
        critical_issues = [i for i in issues if i.level == IssueLevel.CRITICAL]
        error_issues = [i for i in issues if i.level == IssueLevel.ERROR]
        approved = len(critical_issues) == 0 and len(error_issues) <= 2
        
        # Generate summary
        summary = self._generate_summary(issues, data)
        
        return ValidationReport(
            overall_score=overall_score,
            issues=issues,
            summary=summary,
            recommendations=[],  # Will be filled by LLM
            approved=approved
        )
    
    async def _validate_data_basic(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Perform basic data validation checks"""
        
        issues = []
        
        # Check for empty dataset
        if len(data) == 0:
            issues.append(ValidationIssue(
                issue_id="empty_dataset",
                level=IssueLevel.CRITICAL,
                category="data_completeness",
                description="Dataset contains no records",
                affected_data="entire_dataset",
                recommendation="Verify data retrieval query and database connectivity"
            ))
            return issues
        
        # Check for missing required columns
        if 'parameter' in data.columns and 'parameter_value' in data.columns:
            # Check for missing values
            missing_params = data['parameter'].isnull().sum()
            missing_values = data['parameter_value'].isnull().sum()
            
            if missing_params > 0:
                issues.append(ValidationIssue(
                    issue_id="missing_parameters",
                    level=IssueLevel.WARNING,
                    category="data_completeness",
                    description=f"{missing_params} records missing parameter names",
                    affected_data=f"{missing_params}/{len(data)} records",
                    recommendation="Review data extraction process to ensure parameter names are included"
                ))
            
            if missing_values > 0:
                issues.append(ValidationIssue(
                    issue_id="missing_values",
                    level=IssueLevel.WARNING,
                    category="data_completeness",
                    description=f"{missing_values} records missing parameter values",
                    affected_data=f"{missing_values}/{len(data)} records",
                    recommendation="Consider interpolation or exclude incomplete records"
                ))
        
        # Check for duplicate records
        if len(data) > 1:
            duplicate_cols = ['latitude', 'longitude', 'profile_date', 'parameter']
            available_cols = [col for col in duplicate_cols if col in data.columns]
            
            if len(available_cols) >= 2:
                duplicates = data.duplicated(subset=available_cols).sum()
                if duplicates > 0:
                    issues.append(ValidationIssue(
                        issue_id="duplicate_records",
                        level=IssueLevel.WARNING,
                        category="data_quality",
                        description=f"{duplicates} duplicate records detected",
                        affected_data=f"{duplicates}/{len(data)} records",
                        recommendation="Remove duplicates or investigate data source for multiple entries"
                    ))
        
        return issues
    
    async def _validate_data_scientific(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Perform scientific plausibility checks"""
        
        issues = []
        
        if 'parameter' not in data.columns or 'parameter_value' not in data.columns:
            return issues
        
        # Validate each parameter against scientific ranges
        for param in data['parameter'].unique():
            if param in self.parameter_ranges:
                param_data = data[data['parameter'] == param]['parameter_value'].dropna()
                
                if len(param_data) == 0:
                    continue
                
                param_range = self.parameter_ranges[param]
                
                # Check global range violations
                out_of_range = param_data[
                    (param_data < param_range['global_min']) | 
                    (param_data > param_range['global_max'])
                ]
                
                if len(out_of_range) > 0:
                    issues.append(ValidationIssue(
                        issue_id=f"{param}_range_violation",
                        level=IssueLevel.ERROR,
                        category="scientific_validity",
                        description=f"{len(out_of_range)} {param} values outside scientific range "
                                  f"({param_range['global_min']}-{param_range['global_max']} {param_range['units']})",
                        affected_data=f"{len(out_of_range)}/{len(param_data)} {param} values",
                        recommendation=f"Review {param} measurements for sensor calibration or data processing errors"
                    ))
                
                # Check for suspicious values (statistical outliers)
                outliers = self._detect_statistical_outliers(param_data)
                if len(outliers) > 0 and len(outliers) / len(param_data) > 0.05:  # >5% outliers is suspicious
                    issues.append(ValidationIssue(
                        issue_id=f"{param}_statistical_outliers",
                        level=IssueLevel.WARNING,
                        category="statistical_anomaly",
                        description=f"High number of statistical outliers in {param} ({len(outliers)} values)",
                        affected_data=f"{len(outliers)}/{len(param_data)} {param} values",
                        recommendation=f"Investigate {param} outliers for measurement errors or unusual conditions"
                    ))
        
        # Temperature-depth relationship validation
        if all(col in data.columns for col in ['parameter', 'parameter_value', 'pressure']):
            temp_data = data[data['parameter'] == 'TEMP']
            if len(temp_data) > 10:
                depth_temp_issues = self.validation_rules['temperature_depth'](temp_data)
                issues.extend(depth_temp_issues)
        
        return issues
    
    async def _validate_analysis_statistical(self, analysis_results: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate statistical analysis results"""
        
        issues = []
        
        # Validate descriptive statistics
        if 'descriptive' in analysis_results:
            desc_stats = analysis_results['descriptive']
            
            for param, stats in desc_stats.items():
                if isinstance(stats, dict):
                    # Check for impossible statistics
                    if 'std' in stats and stats['std'] < 0:
                        issues.append(ValidationIssue(
                            issue_id=f"{param}_negative_std",
                            level=IssueLevel.ERROR,
                            category="statistical_error",
                            description=f"Negative standard deviation for {param}: {stats['std']}",
                            affected_data=f"{param} statistics",
                            recommendation="Review statistical computation implementation"
                        ))
                    
                    # Check for unrealistic variance
                    if 'mean' in stats and 'std' in stats and stats['mean'] != 0:
                        cv = abs(stats['std'] / stats['mean'])  # Coefficient of variation
                        if cv > 2.0:  # CV > 200% is unusual for oceanographic data
                            issues.append(ValidationIssue(
                                issue_id=f"{param}_high_variance",
                                level=IssueLevel.WARNING,
                                category="statistical_anomaly",
                                description=f"Very high coefficient of variation for {param}: {cv:.2f}",
                                affected_data=f"{param} distribution",
                                recommendation=f"Investigate high variability in {param} - may indicate measurement issues"
                            ))
        
        # Validate trend analysis
        if 'trend' in analysis_results:
            trend_results = analysis_results['trend']
            
            for param, trend in trend_results.items():
                if isinstance(trend, dict) and 'p_value' in trend and 'r_squared' in trend:
                    # Check for claimed significance with low R²
                    if (trend.get('significance') == 'significant' and 
                        trend['p_value'] < 0.05 and 
                        trend['r_squared'] < 0.1):
                        issues.append(ValidationIssue(
                            issue_id=f"{param}_weak_trend",
                            level=IssueLevel.WARNING,
                            category="statistical_interpretation",
                            description=f"Statistically significant but weak trend for {param} (R²={trend['r_squared']:.3f})",
                            affected_data=f"{param} trend analysis",
                            recommendation="Consider practical significance in addition to statistical significance"
                        ))
        
        return issues
    
    async def _validate_cross_consistency(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[ValidationIssue]:
        """Perform cross-validation between different components"""
        
        issues = []
        
        # Validate consistency between raw data and analysis results
        if 'descriptive' in analysis_results and 'parameter' in data.columns:
            for param in data['parameter'].unique():
                if param in analysis_results['descriptive']:
                    param_data = data[data['parameter'] == param]['parameter_value'].dropna()
                    analysis_stats = analysis_results['descriptive'][param]
                    
                    if isinstance(analysis_stats, dict) and len(param_data) > 0:
                        # Check if calculated mean matches
                        actual_mean = param_data.mean()
                        reported_mean = analysis_stats.get('mean')
                        
                        if reported_mean is not None and abs(actual_mean - reported_mean) > 0.01:
                            issues.append(ValidationIssue(
                                issue_id=f"{param}_mean_mismatch",
                                level=IssueLevel.ERROR,
                                category="consistency_error",
                                description=f"Mean mismatch for {param}: raw={actual_mean:.3f}, analysis={reported_mean:.3f}",
                                affected_data=f"{param} statistics",
                                recommendation="Review statistical computation implementation"
                            ))
        
        return issues
    
    def _detect_statistical_outliers(self, data: pd.Series) -> pd.Series:
        """Detect statistical outliers using IQR method"""
        
        if len(data) < 4:
            return pd.Series(dtype=float)
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.stat_thresholds['outlier_iqr_factor'] * IQR
        upper_bound = Q3 + self.stat_thresholds['outlier_iqr_factor'] * IQR
        
        return data[(data < lower_bound) | (data > upper_bound)]
    
    def _validate_temperature_depth_relationship(self, temp_data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate temperature-depth relationship"""
        
        issues = []
        
        if 'pressure' not in temp_data.columns:
            return issues
        
        # Check for temperature inversions (temperature increasing with depth)
        # This is unusual except in specific conditions
        temp_pressure = temp_data[['parameter_value', 'pressure']].dropna().sort_values('pressure')
        
        if len(temp_pressure) > 5:
            temp_diff = temp_pressure['parameter_value'].diff()
            pressure_diff = temp_pressure['pressure'].diff()
            
            # Find where temperature increases significantly with depth
            inversions = temp_diff[(temp_diff > 2.0) & (pressure_diff > 0)]  # >2°C increase with depth
            
            if len(inversions) > 0:
                issues.append(ValidationIssue(
                    issue_id="temperature_inversions",
                    level=IssueLevel.WARNING,
                    category="physical_oceanography",
                    description=f"Temperature inversions detected: {len(inversions)} cases of >2°C increase with depth",
                    affected_data="temperature-depth profile",
                    recommendation="Review for measurement errors or document if thermocline/seasonal effects expected"
                ))
        
        return issues
    
    def _validate_salinity_range(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate salinity values against expected ranges"""
        issues = []
        
        if 'parameter' not in data.columns or 'parameter_value' not in data.columns:
            return issues
        
        sal_data = data[data['parameter'] == 'PSAL']
        if len(sal_data) == 0:
            return issues
        
        param_range = self.parameter_ranges['PSAL']
        values = sal_data['parameter_value'].dropna()
        
        # Global range violations
        out_of_range = values[(values < param_range['global_min']) | (values > param_range['global_max'])]
        if len(out_of_range) > 0:
            issues.append(ValidationIssue(
                issue_id="salinity_global_range_violation",
                level=IssueLevel.ERROR,
                category="scientific_validity",
                description=f"{len(out_of_range)} salinity values outside global range "
                            f"({param_range['global_min']}-{param_range['global_max']} PSU)",
                affected_data=f"{len(out_of_range)}/{len(values)} PSAL values",
                recommendation="Review salinity sensor calibration and data processing steps"
            ))
        
        # Typical open ocean range check
        atypical = values[(values < param_range['typical_min']) | (values > param_range['typical_max'])]
        if len(atypical) / len(values) > 0.2:  # >20% of values are atypical
            issues.append(ValidationIssue(
                issue_id="salinity_atypical_distribution",
                level=IssueLevel.WARNING,
                category="scientific_validity",
                description=f"Large fraction of salinity values ({len(atypical)}/{len(values)}) outside typical open ocean range "
                            f"({param_range['typical_min']}-{param_range['typical_max']} PSU)",
                affected_data="PSAL distribution",
                recommendation="Verify if dataset includes coastal or estuarine waters where salinity is expected to vary"
            ))
        
        return issues

    def _validate_pressure_consistency(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate pressure (depth) measurements for monotonicity and physical limits"""
        issues = []
        
        if 'pressure' not in data.columns:
            return issues
        
        values = data['pressure'].dropna()
        if len(values) == 0:
            return issues
        
        # Global max depth
        max_depth = self.parameter_ranges['PRES']['global_max']
        invalid = values[values > max_depth]
        if len(invalid) > 0:
            issues.append(ValidationIssue(
                issue_id="pressure_exceeds_max",
                level=IssueLevel.ERROR,
                category="scientific_validity",
                description=f"{len(invalid)} pressure values exceed maximum ocean depth ({max_depth} dbar)",
                affected_data=f"{len(invalid)} pressure readings",
                recommendation="Check unit consistency (e.g., Pa vs dbar) and data source validity"
            ))
        
        # Monotonicity within profile (if profile_date exists)
        if 'profile_date' in data.columns:
            grouped = data.groupby('profile_date')
            for ts, grp in grouped:
                sorted_grp = grp.sort_values('pressure')
                if not sorted_grp['pressure'].is_monotonic_increasing:
                    issues.append(ValidationIssue(
                        issue_id=f"non_monotonic_pressure_{ts}",
                        level=IssueLevel.WARNING,
                        category="data_quality",
                        description=f"Non-monotonic pressure profile detected at {ts}",
                        affected_data=f"{len(grp)} records in profile {ts}",
                        recommendation="Ensure proper ordering of depth profiles"
                    ))
        
        return issues

    def _validate_temporal_consistency(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Check for temporal consistency in time series data"""
        issues = []
        
        if 'profile_date' not in data.columns:
            return issues
        
        # Parse to datetime if needed
        try:
            times = pd.to_datetime(data['profile_date'], errors='coerce').dropna()
        except Exception:
            return issues
        
        if len(times) < 2:
            return issues
        
        diffs = times.sort_values().diff().dropna()
        
        # Look for unrealistic time gaps (e.g., duplicates or gaps > 1 year)
        if (diffs == pd.Timedelta(0)).any():
            issues.append(ValidationIssue(
                issue_id="duplicate_timestamps",
                level=IssueLevel.WARNING,
                category="temporal_consistency",
                description="Duplicate timestamps detected in profile data",
                affected_data="profile_date",
                recommendation="Verify time recording process and deduplicate entries"
            ))
        
        if (diffs > pd.Timedelta(days=365)).any():
            issues.append(ValidationIssue(
                issue_id="large_temporal_gap",
                level=IssueLevel.INFO,
                category="temporal_consistency",
                description="Unusually large gaps (>1 year) between consecutive observations",
                affected_data="profile_date",
                recommendation="Confirm data continuity; large gaps may reflect incomplete records"
            ))
        
        return issues

    def _validate_spatial_consistency(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Check for spatial consistency (lat/lon ranges, duplicates, clustering)"""
        issues = []
        
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            return issues
        
        lat = data['latitude'].dropna()
        lon = data['longitude'].dropna()
        
        # Range check
        if ((lat < -90) | (lat > 90)).any() or ((lon < -180) | (lon > 180)).any():
            issues.append(ValidationIssue(
                issue_id="invalid_coordinates",
                level=IssueLevel.ERROR,
                category="spatial_consistency",
                description="Detected latitude/longitude values outside valid ranges",
                affected_data="latitude/longitude",
                recommendation="Verify coordinate system and units"
            ))
        
        # Check for clustering (all points nearly identical)
        if len(lat) > 10 and lat.std() < 1e-4 and lon.std() < 1e-4:
            issues.append(ValidationIssue(
                issue_id="spatial_clustering",
                level=IssueLevel.WARNING,
                category="spatial_consistency",
                description="Most observations located at nearly identical coordinates",
                affected_data="latitude/longitude",
                recommendation="Check if dataset represents a fixed mooring; if not, review location recording"
            ))
        
        return issues

    def _validate_statistical_results(self, analysis_results: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate higher-level statistical analysis outputs"""
        issues = []
        
        if not analysis_results:
            return issues
        
        # Correlation matrix validation
        if 'correlations' in analysis_results and isinstance(analysis_results['correlations'], dict):
            for pair, corr in analysis_results['correlations'].items():
                if abs(corr) > self.stat_thresholds['correlation_threshold']:
                    issues.append(ValidationIssue(
                        issue_id=f"high_correlation_{pair}",
                        level=IssueLevel.INFO,
                        category="statistical_validity",
                        description=f"Very high correlation ({corr:.2f}) detected between {pair}",
                        affected_data="correlation_matrix",
                        recommendation="Check for redundancy or multicollinearity in analysis"
                    ))
        
        # Sample size adequacy
        if 'sample_sizes' in analysis_results and isinstance(analysis_results['sample_sizes'], dict):
            for param, n in analysis_results['sample_sizes'].items():
                if n < self.stat_thresholds['min_sample_size']:
                    issues.append(ValidationIssue(
                        issue_id=f"insufficient_sample_{param}",
                        level=IssueLevel.WARNING,
                        category="statistical_validity",
                        description=f"Insufficient sample size for {param}: n={n}, minimum required {self.stat_thresholds['min_sample_size']}",
                        affected_data=f"{param} sample",
                        recommendation="Collect more data before drawing statistical conclusions"
                    ))
        
        return issues

    
    def _calculate_overall_score(self, issues: List[ValidationIssue], data_size: int) -> float:
        """Calculate overall quality score (0-100)"""
        
        if data_size == 0:
            return 0.0
        
        # Scoring weights
        weights = {
            IssueLevel.CRITICAL: 25,
            IssueLevel.ERROR: 10,
            IssueLevel.WARNING: 3,
            IssueLevel.INFO: 1
        }
        
        # Calculate penalty score
        penalty = sum(weights[issue.level] for issue in issues)
        
        # Scale penalty relative to data size
        max_possible_penalty = data_size * weights[IssueLevel.WARNING]  # Assume max 1 warning per record
        normalized_penalty = min(penalty, max_possible_penalty)
        
        # Calculate score (100 - penalty percentage)
        score = max(0.0, 100.0 - (normalized_penalty / max_possible_penalty * 100.0))
        
        return round(score, 1)
    
    def _generate_summary(self, issues: List[ValidationIssue], data: pd.DataFrame) -> Dict[str, Any]:
        """Generate validation summary"""
        
        issue_counts = {level.value: 0 for level in IssueLevel}
        for issue in issues:
            issue_counts[issue.level.value] += 1
        
        categories = {}
        for issue in issues:
            if issue.category not in categories:
                categories[issue.category] = 0
            categories[issue.category] += 1
        
        return {
            'total_records': len(data),
            'total_issues': len(issues),
            'issue_counts': issue_counts,
            'categories': categories,
            'top_issues': [issue.description for issue in sorted(issues, 
                          key=lambda x: ['info', 'warning', 'error', 'critical'].index(x.level.value), 
                          reverse=True)[:3]]
        }
    
    async def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate additional recommendations using LLM"""
        
        # Create summary for LLM
        summary_data = {
            'overall_score': report.overall_score,
            'total_issues': len(report.issues),
            'critical_issues': len([i for i in report.issues if i.level == IssueLevel.CRITICAL]),
            'error_issues': len([i for i in report.issues if i.level == IssueLevel.ERROR]),
            'top_issues': report.summary.get('top_issues', []),
            'approved': report.approved
        }
        
        system_prompt = """You are an oceanographic data quality expert. Based on validation results, 
        provide 3-5 specific recommendations for improving data quality or analysis.
        
        Focus on:
        1. Data quality improvements
        2. Analysis methodology suggestions  
        3. Scientific interpretation guidance
        4. Next steps for investigation
        
        Be specific and actionable."""
        
        user_prompt = f"Provide recommendations based on this validation summary: {json.dumps(summary_data)}"
        
        try:
            messages = [
                self.create_system_message(system_prompt),
                self.create_user_message(user_prompt)
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            # Parse recommendations
            recommendations = [line.strip() for line in response.content.split('\n') 
                             if line.strip() and not line.strip().isdigit()]
            
            return recommendations[:5]
            
        except Exception as e:
            self.logger.warning(f"Failed to generate recommendations with LLM: {e}")
            return ["Review validation issues and implement suggested corrections."]
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, report: ValidationReport) -> Dict[str, Any]:
        """Calculate additional quality metrics"""
        
        metrics = {
            'completeness': 1.0,
            'accuracy': 1.0,
            'consistency': 1.0,
            'validity': 1.0
        }
        
        if len(data) > 0:
            # Completeness: proportion of non-null values
            if 'parameter_value' in data.columns:
                completeness = 1.0 - (data['parameter_value'].isnull().sum() / len(data))
                metrics['completeness'] = completeness
            
            # Other metrics based on validation issues
            error_weight = len([i for i in report.issues if i.level in [IssueLevel.ERROR, IssueLevel.CRITICAL]])
            warning_weight = len([i for i in report.issues if i.level == IssueLevel.WARNING])
            
            total_penalty = error_weight * 0.1 + warning_weight * 0.05
            adjustment = max(0.0, 1.0 - total_penalty)
            
            metrics['accuracy'] = adjustment
            metrics['consistency'] = adjustment  
            metrics['validity'] = adjustment
        
        return metrics
    
    def _issue_to_dict(self, issue: ValidationIssue) -> Dict[str, Any]:
        """Convert ValidationIssue to dictionary"""
        return {
            'id': issue.issue_id,
            'level': issue.level.value,
            'category': issue.category,
            'description': issue.description,
            'affected_data': issue.affected_data,
            'recommendation': issue.recommendation,
            'metadata': issue.metadata or {}
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about validation capabilities"""
        return {
            "description": "Validates oceanographic data and analysis results for scientific accuracy",
            "validation_levels": [level.value for level in ValidationLevel],
            "issue_levels": [level.value for level in IssueLevel],
            "parameter_ranges": list(self.parameter_ranges.keys()),
            "validation_categories": ["data_completeness", "data_quality", "scientific_validity", "statistical_validity"]
        }