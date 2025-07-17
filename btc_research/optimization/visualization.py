"""
Visualization and reporting module for optimization results.

This module provides comprehensive visualization capabilities for optimization
results, including parameter space exploration, convergence plots, validation
metrics, and robustness analysis.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import plotting libraries with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Visualization features disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from btc_research.optimization.types import (
    OptimizationResult,
    ValidationResult,
    RobustnessResult,
    StatisticsResult,
)

__all__ = [
    "OptimizationVisualizer",
    "create_optimization_report",
    "plot_convergence",
    "plot_parameter_importance",
    "plot_validation_metrics",
    "plot_robustness_analysis",
]


class OptimizationVisualizer:
    """
    Comprehensive visualization toolkit for optimization results.
    
    This class provides methods to create various plots and reports
    for optimization analysis including convergence, parameter importance,
    validation metrics, and robustness testing results.
    """
    
    def __init__(
        self,
        results: Dict[str, Any],
        style: str = "seaborn",
        output_dir: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ):
        """
        Initialize optimization visualizer.
        
        Args:
            results: Complete optimization results dictionary
            style: Plotting style ("seaborn", "ggplot", "default")
            output_dir: Directory to save plots (if None, displays only)
            interactive: Use interactive plots (plotly) when available
        """
        self.results = results
        self.style = style
        self.output_dir = Path(output_dir) if output_dir else None
        self.interactive = interactive and PLOTLY_AVAILABLE
        
        # Create output directory if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        if PLOTTING_AVAILABLE:
            plt.style.use(style if style in plt.style.available else 'default')
            sns.set_palette("husl")
    
    def create_full_report(self) -> Dict[str, str]:
        """
        Create comprehensive optimization report with all visualizations.
        
        Returns:
            Dictionary mapping plot names to file paths (if saved)
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping visualization.")
            return {}
        
        plots_created = {}
        
        # 1. Convergence plot
        try:
            convergence_path = self.plot_convergence()
            if convergence_path:
                plots_created["convergence"] = convergence_path
        except Exception as e:
            print(f"Warning: Could not create convergence plot: {e}")
        
        # 2. Parameter importance plot
        try:
            importance_path = self.plot_parameter_importance()
            if importance_path:
                plots_created["parameter_importance"] = importance_path
        except Exception as e:
            print(f"Warning: Could not create parameter importance plot: {e}")
        
        # 3. Validation metrics plot
        try:
            validation_path = self.plot_validation_metrics()
            if validation_path:
                plots_created["validation_metrics"] = validation_path
        except Exception as e:
            print(f"Warning: Could not create validation metrics plot: {e}")
        
        # 4. Robustness analysis plot
        try:
            robustness_path = self.plot_robustness_analysis()
            if robustness_path:
                plots_created["robustness_analysis"] = robustness_path
        except Exception as e:
            print(f"Warning: Could not create robustness analysis plot: {e}")
        
        # 5. Parameter correlation heatmap
        try:
            correlation_path = self.plot_parameter_correlation()
            if correlation_path:
                plots_created["parameter_correlation"] = correlation_path
        except Exception as e:
            print(f"Warning: Could not create parameter correlation plot: {e}")
        
        # 6. Performance distribution plot
        try:
            distribution_path = self.plot_performance_distribution()
            if distribution_path:
                plots_created["performance_distribution"] = distribution_path
        except Exception as e:
            print(f"Warning: Could not create performance distribution plot: {e}")
        
        return plots_created
    
    def plot_convergence(self, save_name: str = "convergence_plot") -> Optional[str]:
        """
        Plot optimization convergence over iterations.
        
        Args:
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot file or None
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        opt_result = self.results.get("optimization")
        if not opt_result or not hasattr(opt_result, 'diagnostics'):
            print("No convergence data available for plotting")
            return None
        
        # Extract convergence data from diagnostics
        diagnostics = opt_result.diagnostics
        if "iteration_history" not in diagnostics:
            print("No iteration history available for convergence plot")
            return None
        
        iteration_history = diagnostics["iteration_history"]
        iterations = list(range(len(iteration_history)))
        
        if self.interactive:
            # Create interactive plot with plotly
            fig = go.Figure()
            
            # Best score over iterations
            best_scores = []
            current_best = float('-inf')
            for score in iteration_history:
                if score > current_best:
                    current_best = score
                best_scores.append(current_best)
            
            fig.add_trace(go.Scatter(
                x=iterations,
                y=iteration_history,
                mode='markers',
                name='Iteration Score',
                opacity=0.6,
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=iterations,
                y=best_scores,
                mode='lines',
                name='Best Score',
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title="Optimization Convergence",
                xaxis_title="Iteration",
                yaxis_title="Objective Value",
                hovermode='x unified'
            )
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.html"
                fig.write_html(save_path)
                return str(save_path)
            else:
                fig.show()
                return None
        
        else:
            # Create static plot with matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot iteration scores
            ax.scatter(iterations, iteration_history, alpha=0.6, label='Iteration Score', s=30)
            
            # Plot best score line
            best_scores = []
            current_best = float('-inf')
            for score in iteration_history:
                if score > current_best:
                    current_best = score
                best_scores.append(current_best)
            
            ax.plot(iterations, best_scores, 'r-', linewidth=2, label='Best Score')
            
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Objective Value")
            ax.set_title("Optimization Convergence")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(save_path)
            else:
                plt.show()
                return None
    
    def plot_parameter_importance(self, save_name: str = "parameter_importance") -> Optional[str]:
        """
        Plot parameter importance based on optimization results.
        
        Args:
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot file or None
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        # Extract parameter importance from diagnostics
        opt_result = self.results.get("optimization")
        if not opt_result or not hasattr(opt_result, 'diagnostics'):
            return None
        
        diagnostics = opt_result.diagnostics
        if "parameter_importance" not in diagnostics:
            print("No parameter importance data available")
            return None
        
        importance_data = diagnostics["parameter_importance"]
        params = list(importance_data.keys())
        importance_scores = list(importance_data.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_params = [params[i] for i in sorted_indices]
        sorted_scores = [importance_scores[i] for i in sorted_indices]
        
        if self.interactive:
            # Create interactive bar plot
            fig = go.Figure(data=[
                go.Bar(x=sorted_scores, y=sorted_params, orientation='h')
            ])
            
            fig.update_layout(
                title="Parameter Importance",
                xaxis_title="Importance Score",
                yaxis_title="Parameters",
                height=400 + len(params) * 20
            )
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.html"
                fig.write_html(save_path)
                return str(save_path)
            else:
                fig.show()
                return None
        
        else:
            # Create static bar plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.5)))
            
            bars = ax.barh(sorted_params, sorted_scores)
            ax.set_xlabel("Importance Score")
            ax.set_title("Parameter Importance")
            
            # Color bars by importance level
            max_score = max(sorted_scores)
            for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
                bar.set_color(plt.cm.viridis(score / max_score))
            
            plt.tight_layout()
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(save_path)
            else:
                plt.show()
                return None
    
    def plot_validation_metrics(self, save_name: str = "validation_metrics") -> Optional[str]:
        """
        Plot validation metrics across folds.
        
        Args:
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot file or None
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        validation_result = self.results.get("validation")
        if not validation_result or not validation_result.fold_results:
            print("No validation results available for plotting")
            return None
        
        # Create DataFrame from fold results
        df = pd.DataFrame(validation_result.fold_results)
        metrics = df.columns.tolist()
        
        if self.interactive:
            # Create interactive box plots
            fig = make_subplots(
                rows=1, cols=len(metrics),
                subplot_titles=metrics,
                horizontal_spacing=0.1
            )
            
            for i, metric in enumerate(metrics, 1):
                fig.add_trace(
                    go.Box(y=df[metric], name=metric, showlegend=False),
                    row=1, col=i
                )
            
            fig.update_layout(
                title="Validation Metrics Across Folds",
                height=500
            )
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.html"
                fig.write_html(save_path)
                return str(save_path)
            else:
                fig.show()
                return None
        
        else:
            # Create static box plots
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
            
            if n_metrics == 1:
                axes = [axes]
            
            for ax, metric in zip(axes, metrics):
                ax.boxplot(df[metric])
                ax.set_title(metric)
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
            
            plt.suptitle("Validation Metrics Across Folds")
            plt.tight_layout()
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(save_path)
            else:
                plt.show()
                return None
    
    def plot_robustness_analysis(self, save_name: str = "robustness_analysis") -> Optional[str]:
        """
        Plot robustness testing results.
        
        Args:
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot file or None
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        robustness_result = self.results.get("robustness")
        if not robustness_result:
            print("No robustness results available for plotting")
            return None
        
        # Extract simulation results
        simulation_results = robustness_result.results
        if not simulation_results:
            return None
        
        # Create DataFrame from simulation results
        df = pd.DataFrame(simulation_results)
        
        # Plot distribution of primary metric
        primary_metric = next(iter(df.columns))
        
        if self.interactive:
            # Create interactive histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df[primary_metric],
                nbinsx=50,
                name="Simulation Results",
                opacity=0.7
            ))
            
            # Add VaR and ES lines if available
            if hasattr(robustness_result, 'value_at_risk'):
                var_95 = robustness_result.value_at_risk.get('95%', None)
                if var_95:
                    fig.add_vline(x=var_95, line_dash="dash", line_color="red",
                                annotation_text="VaR 95%")
            
            fig.update_layout(
                title=f"Robustness Analysis - {primary_metric} Distribution",
                xaxis_title=primary_metric,
                yaxis_title="Frequency"
            )
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.html"
                fig.write_html(save_path)
                return str(save_path)
            else:
                fig.show()
                return None
        
        else:
            # Create static histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(df[primary_metric], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(primary_metric)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Robustness Analysis - {primary_metric} Distribution")
            
            # Add VaR line if available
            if hasattr(robustness_result, 'value_at_risk'):
                var_95 = robustness_result.value_at_risk.get('95%', None)
                if var_95:
                    ax.axvline(var_95, color='red', linestyle='--', linewidth=2, label='VaR 95%')
                    ax.legend()
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(save_path)
            else:
                plt.show()
                return None
    
    def plot_parameter_correlation(self, save_name: str = "parameter_correlation") -> Optional[str]:
        """
        Plot correlation matrix of optimization parameters.
        
        Args:
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot file or None
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        opt_result = self.results.get("optimization")
        if not opt_result or not hasattr(opt_result, 'diagnostics'):
            return None
        
        diagnostics = opt_result.diagnostics
        if "parameter_history" not in diagnostics:
            print("No parameter history available for correlation analysis")
            return None
        
        # Create DataFrame from parameter history
        param_history = diagnostics["parameter_history"]
        df = pd.DataFrame(param_history)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        if self.interactive:
            # Create interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Parameter Correlation Matrix",
                width=600,
                height=600
            )
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.html"
                fig.write_html(save_path)
                return str(save_path)
            else:
                fig.show()
                return None
        
        else:
            # Create static heatmap
            fig, ax = plt.subplots(figsize=(8, 8))
            
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title("Parameter Correlation Matrix")
            
            plt.tight_layout()
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(save_path)
            else:
                plt.show()
                return None
    
    def plot_performance_distribution(self, save_name: str = "performance_distribution") -> Optional[str]:
        """
        Plot performance metric distributions.
        
        Args:
            save_name: Name for saved plot file
            
        Returns:
            Path to saved plot file or None
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        # Combine data from validation and robustness testing
        data_sources = []
        
        validation_result = self.results.get("validation")
        if validation_result and validation_result.fold_results:
            val_df = pd.DataFrame(validation_result.fold_results)
            val_df['source'] = 'Validation'
            data_sources.append(val_df)
        
        robustness_result = self.results.get("robustness")
        if robustness_result and robustness_result.results:
            rob_df = pd.DataFrame(robustness_result.results)
            rob_df['source'] = 'Robustness'
            data_sources.append(rob_df)
        
        if not data_sources:
            print("No performance data available for distribution plot")
            return None
        
        # Combine data
        combined_df = pd.concat(data_sources, ignore_index=True)
        
        # Get numeric columns only
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'source' in numeric_cols:
            numeric_cols.remove('source')
        
        if not numeric_cols:
            return None
        
        # Plot distributions for each metric
        if self.interactive:
            # Create subplots for each metric
            n_metrics = len(numeric_cols)
            fig = make_subplots(
                rows=(n_metrics + 1) // 2, cols=2,
                subplot_titles=numeric_cols
            )
            
            for i, metric in enumerate(numeric_cols):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                for source in combined_df['source'].unique():
                    source_data = combined_df[combined_df['source'] == source][metric]
                    fig.add_trace(
                        go.Histogram(x=source_data, name=f"{source} - {metric}",
                                   opacity=0.7, showlegend=(i == 0)),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title="Performance Metric Distributions",
                height=300 * ((n_metrics + 1) // 2)
            )
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.html"
                fig.write_html(save_path)
                return str(save_path)
            else:
                fig.show()
                return None
        
        else:
            # Create static violin plots
            n_metrics = len(numeric_cols)
            n_cols = 2
            n_rows = (n_metrics + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
            
            for i, metric in enumerate(numeric_cols):
                ax = axes[i]
                
                # Create violin plot
                data_by_source = [combined_df[combined_df['source'] == source][metric] 
                                for source in combined_df['source'].unique()]
                
                parts = ax.violinplot(data_by_source, positions=range(len(data_by_source)))
                
                ax.set_title(metric)
                ax.set_xticks(range(len(combined_df['source'].unique())))
                ax.set_xticklabels(combined_df['source'].unique())
                ax.grid(True, alpha=0.3)
            
            # Hide extra subplots
            for j in range(len(numeric_cols), len(axes)):
                axes[j].set_visible(False)
            
            plt.suptitle("Performance Metric Distributions")
            plt.tight_layout()
            
            if self.output_dir:
                save_path = self.output_dir / f"{save_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(save_path)
            else:
                plt.show()
                return None


def create_optimization_report(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    title: str = "Optimization Report",
    include_plots: bool = True,
    interactive: bool = False,
) -> str:
    """
    Create comprehensive HTML optimization report.
    
    Args:
        results: Complete optimization results
        output_dir: Directory to save report
        title: Report title
        include_plots: Whether to include visualizations
        interactive: Use interactive plots
        
    Returns:
        Path to generated HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations if requested
    plots = {}
    if include_plots:
        visualizer = OptimizationVisualizer(
            results, output_dir=output_dir, interactive=interactive
        )
        plots = visualizer.create_full_report()
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .section {{ margin: 30px 0; }}
        .metric {{ margin: 10px 0; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        {_generate_summary_html(results)}
    </div>
    
    <div class="section">
        <h2>Optimization Results</h2>
        {_generate_optimization_html(results)}
    </div>
    
    <div class="section">
        <h2>Validation Analysis</h2>
        {_generate_validation_html(results)}
    </div>
    
    <div class="section">
        <h2>Robustness Testing</h2>
        {_generate_robustness_html(results)}
    </div>
    
    {_generate_plots_html(plots) if plots else ""}
    
</body>
</html>
    """
    
    # Save HTML report
    report_path = output_dir / "optimization_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return str(report_path)


def _generate_summary_html(results: Dict[str, Any]) -> str:
    """Generate executive summary HTML section."""
    summary = results.get("summary", {})
    
    html = "<ul>"
    html += f"<li><strong>Best Objective Value:</strong> {summary.get('best_objective_value', 'N/A'):.4f}</li>"
    html += f"<li><strong>Optimization Method:</strong> {results.get('framework_info', {}).get('optimizer', 'Unknown')}</li>"
    html += f"<li><strong>Validation Strategy:</strong> {results.get('framework_info', {}).get('validator', 'Unknown')}</li>"
    html += f"<li><strong>Total Duration:</strong> {results.get('framework_info', {}).get('total_duration', 0):.1f} seconds</li>"
    
    if 'potential_overfitting' in summary:
        html += f"<li><strong>Overfitting Risk:</strong> {'High' if summary['potential_overfitting'] else 'Low'}</li>"
    
    html += "</ul>"
    return html


def _generate_optimization_html(results: Dict[str, Any]) -> str:
    """Generate optimization results HTML section."""
    opt_result = results.get("optimization")
    if not opt_result:
        return "<p>No optimization results available.</p>"
    
    html = "<h3>Best Parameters</h3><table><tr><th>Parameter</th><th>Value</th></tr>"
    for param, value in opt_result.parameters.items():
        html += f"<tr><td>{param}</td><td>{value}</td></tr>"
    html += "</table>"
    
    html += "<h3>Performance Metrics</h3><table><tr><th>Metric</th><th>Value</th></tr>"
    for metric, value in opt_result.metrics.items():
        html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
    html += "</table>"
    
    return html


def _generate_validation_html(results: Dict[str, Any]) -> str:
    """Generate validation analysis HTML section."""
    validation_result = results.get("validation")
    if not validation_result:
        return "<p>No validation results available.</p>"
    
    html = f"<p><strong>Method:</strong> {validation_result.method.value}</p>"
    html += f"<p><strong>Number of Folds:</strong> {validation_result.n_splits}</p>"
    html += f"<p><strong>Stability Score:</strong> {validation_result.stability_score:.4f}</p>"
    
    if validation_result.mean_metrics:
        html += "<h3>Cross-Validation Metrics</h3><table><tr><th>Metric</th><th>Mean</th><th>Std</th></tr>"
        for metric, mean_val in validation_result.mean_metrics.items():
            std_val = validation_result.std_metrics.get(metric, 0)
            html += f"<tr><td>{metric}</td><td>{mean_val:.4f}</td><td>{std_val:.4f}</td></tr>"
        html += "</table>"
    
    return html


def _generate_robustness_html(results: Dict[str, Any]) -> str:
    """Generate robustness testing HTML section."""
    robustness_result = results.get("robustness")
    if not robustness_result:
        return "<p>No robustness testing results available.</p>"
    
    html = f"<p><strong>Test Type:</strong> {robustness_result.test_type}</p>"
    html += f"<p><strong>Simulations:</strong> {robustness_result.n_simulations}</p>"
    html += f"<p><strong>Success Rate:</strong> {robustness_result.success_rate:.1%}</p>"
    
    return html


def _generate_plots_html(plots: Dict[str, str]) -> str:
    """Generate plots HTML section."""
    if not plots:
        return ""
    
    html = '<div class="section"><h2>Visualizations</h2>'
    
    for plot_name, plot_path in plots.items():
        plot_title = plot_name.replace('_', ' ').title()
        
        if plot_path.endswith('.html'):
            # Embed interactive plot
            html += f'<div class="plot"><h3>{plot_title}</h3>'
            html += f'<iframe src="{Path(plot_path).name}" width="100%" height="500px"></iframe></div>'
        else:
            # Embed static image
            html += f'<div class="plot"><h3>{plot_title}</h3>'
            html += f'<img src="{Path(plot_path).name}" style="max-width: 100%; height: auto;"></div>'
    
    html += '</div>'
    return html


# Convenience functions for individual plots
def plot_convergence(results: Dict[str, Any], **kwargs) -> Optional[str]:
    """Plot optimization convergence."""
    visualizer = OptimizationVisualizer(results, **kwargs)
    return visualizer.plot_convergence()


def plot_parameter_importance(results: Dict[str, Any], **kwargs) -> Optional[str]:
    """Plot parameter importance."""
    visualizer = OptimizationVisualizer(results, **kwargs)
    return visualizer.plot_parameter_importance()


def plot_validation_metrics(results: Dict[str, Any], **kwargs) -> Optional[str]:
    """Plot validation metrics."""
    visualizer = OptimizationVisualizer(results, **kwargs)
    return visualizer.plot_validation_metrics()


def plot_robustness_analysis(results: Dict[str, Any], **kwargs) -> Optional[str]:
    """Plot robustness analysis."""
    visualizer = OptimizationVisualizer(results, **kwargs)
    return visualizer.plot_robustness_analysis()