#!/usr/bin/env python3
"""
Volume Profile Strategy Analysis & Improvement Tool

This script analyzes your current volume profile strategy performance
and provides detailed insights on the proposed improvements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import yaml

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StrategyAnalyzer:
    """Analyze and compare volume profile trading strategies."""
    
    def __init__(self):
        self.results = {}
        
    def load_backtest_results(self, period: str, total_return: float, 
                            sharpe_ratio: float, max_drawdown: float,
                            num_trades: int, win_rate: float, profit_factor: float):
        """Load backtest results for analysis."""
        self.results[period] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': total_return / num_trades if num_trades > 0 else 0
        }
    
    def analyze_current_performance(self):
        """Analyze current strategy performance across different market regimes."""
        print("="*80)
        print("CURRENT VOLUME PROFILE STRATEGY ANALYSIS")
        print("="*80)
        
        # Load your provided results
        self.load_backtest_results("2021-2025 (Full)", 171.61, 0.90, 40.72, 332, 33.43, 1.35)
        self.load_backtest_results("2021-11 to 2022-11 (Bear)", 4.03, 0.16, 24.65, 108, 27.78, 1.03)
        self.load_backtest_results("2022-11 to 2025-01 (Bull)", 136.01, 0.66, 24.37, 174, 35.06, 1.65)
        
        # Create performance summary
        df = pd.DataFrame(self.results).T
        
        print("\nPERFORMANCE SUMMARY:")
        print("-" * 50)
        for period, metrics in self.results.items():
            print(f"\n{period}:")
            print(f"  Total Return:     {metrics['total_return']:>8.2f}%")
            print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>8.2f}")
            print(f"  Max Drawdown:     {metrics['max_drawdown']:>8.2f}%")
            print(f"  Win Rate:         {metrics['win_rate']:>8.2f}%")
            print(f"  Profit Factor:    {metrics['profit_factor']:>8.2f}")
            print(f"  Trades:           {metrics['num_trades']:>8}")
        
        return df
    
    def identify_weaknesses(self):
        """Identify key weaknesses in the current strategy."""
        print("\n" + "="*80)
        print("IDENTIFIED WEAKNESSES & ROOT CAUSES")
        print("="*80)
        
        weaknesses = [
            {
                'issue': 'Poor Bear Market Performance',
                'evidence': 'Sharpe ratio of 0.16 vs 0.66 in bull markets',
                'impact': 'Strategy struggles in ranging/declining markets',
                'root_cause': 'Trend-following bias without regime adaptation'
            },
            {
                'issue': 'Low Win Rate (~30%)',
                'evidence': 'Consistent across all periods (27.78% - 35.06%)',
                'impact': 'Psychological difficulty, large losing streaks',
                'root_cause': 'Insufficient confluence, premature entries'
            },
            {
                'issue': 'High Maximum Drawdown',
                'evidence': '40.72% max drawdown in full period',
                'impact': 'Unacceptable for most risk tolerances',
                'root_cause': 'Inadequate position sizing and stop losses'
            },
            {
                'issue': 'Market Regime Blindness',
                'evidence': 'No adaptation between bull/bear periods',
                'impact': 'Suboptimal performance across cycles',
                'root_cause': 'Single strategy approach for all conditions'
            }
        ]
        
        for i, weakness in enumerate(weaknesses, 1):
            print(f"\n{i}. {weakness['issue']}")
            print(f"   Evidence:    {weakness['evidence']}")
            print(f"   Impact:      {weakness['impact']}")
            print(f"   Root Cause:  {weakness['root_cause']}")
    
    def propose_improvements(self):
        """Propose specific improvements based on analysis."""
        print("\n" + "="*80)
        print("PROPOSED STRATEGY IMPROVEMENTS")
        print("="*80)
        
        improvements = [
            {
                'strategy': 'Improved Volume Profile V1',
                'focus': 'Enhanced Risk Management & Market Regime Filtering',
                'key_changes': [
                    'Market regime detection using 4H ADX',
                    'Multi-timeframe EMA trend filtering',
                    'Reduced risk per trade (1.5% vs current)',
                    'Dynamic stops based on market conditions',
                    'Partial profit taking for better win rate'
                ],
                'expected_improvements': [
                    'Better bear market performance',
                    'Reduced maximum drawdown',
                    'More consistent returns across regimes'
                ]
            },
            {
                'strategy': 'Simplified VP-FVG Strategy',
                'focus': 'Clean confluence without overcomplications',
                'key_changes': [
                    'Simplified FVG + VP + trend confluence',
                    'Removed complex VPFVGSignal dependency',
                    'Clear entry/exit conditions',
                    'Conservative 2:1 risk-reward ratio'
                ],
                'expected_improvements': [
                    'Higher win rate through better confluence',
                    'Easier to understand and trade',
                    'More reliable signals'
                ]
            },
            {
                'strategy': 'Bear Market Optimized VP',
                'focus': 'Specialized for bear/sideways markets',
                'key_changes': [
                    'Mean reversion focus in ranging markets',
                    'Conservative position sizing (1% risk)',
                    'Quick profit taking (3% targets)',
                    'Tight stops (2%) for capital preservation'
                ],
                'expected_improvements': [
                    'Positive returns in bear markets',
                    'Higher Sharpe ratio in difficult conditions',
                    'Reduced correlation with market direction'
                ]
            },
            {
                'strategy': 'Multi-Timeframe VP Enhanced',
                'focus': 'Improved win rate through better confluence',
                'key_changes': [
                    'Triple timeframe volume profile analysis',
                    'Multi-timeframe trend alignment requirements',
                    'Staged profit taking for higher win rates',
                    'Quality filters to avoid low-probability trades'
                ],
                'expected_improvements': [
                    'Win rate improvement to 40-50%+',
                    'Better risk-adjusted returns',
                    'More selective, higher-quality trades'
                ]
            }
        ]
        
        for improvement in improvements:
            print(f"\n📈 {improvement['strategy']}")
            print(f"Focus: {improvement['focus']}")
            
            print("\nKey Changes:")
            for change in improvement['key_changes']:
                print(f"  • {change}")
            
            print("\nExpected Improvements:")
            for exp in improvement['expected_improvements']:
                print(f"  ✓ {exp}")
    
    def risk_management_recommendations(self):
        """Provide specific risk management recommendations."""
        print("\n" + "="*80)
        print("RISK MANAGEMENT RECOMMENDATIONS")
        print("="*80)
        
        recommendations = [
            {
                'area': 'Position Sizing',
                'current_issue': 'Fixed position sizes regardless of market conditions',
                'recommendation': 'Dynamic sizing based on volatility and regime',
                'implementation': [
                    'Use 1-2% risk per trade (vs current higher amounts)',
                    'Scale down in volatile/uncertain markets',
                    'Scale up in trending markets with higher win rates'
                ]
            },
            {
                'area': 'Stop Loss Management',
                'current_issue': 'Static stops not adapted to market structure',
                'recommendation': 'Volume Profile-based dynamic stops',
                'implementation': [
                    'Use VP levels (VAL/VAH) as natural stop levels',
                    'Add small buffer (0.2%) beyond VP levels',
                    'Tighter stops in ranging markets, wider in trends'
                ]
            },
            {
                'area': 'Profit Taking',
                'current_issue': 'Binary exit approach hurts win rate',
                'recommendation': 'Staged profit taking strategy',
                'implementation': [
                    'Take 30-40% profits at 2% gain',
                    'Take another 30-40% at 3.5% gain',
                    'Let remaining position run to major targets'
                ]
            },
            {
                'area': 'Market Regime Adaptation',
                'current_issue': 'Same strategy regardless of market conditions',
                'recommendation': 'Regime-specific parameter adjustments',
                'implementation': [
                    'Bear markets: Mean reversion, tight stops, quick profits',
                    'Bull markets: Trend following, wider stops, larger targets',
                    'Ranging markets: Support/resistance plays, neutral bias'
                ]
            }
        ]
        
        for rec in recommendations:
            print(f"\n🎯 {rec['area']}")
            print(f"Current Issue: {rec['current_issue']}")
            print(f"Recommendation: {rec['recommendation']}")
            print("Implementation:")
            for impl in rec['implementation']:
                print(f"  • {impl}")
    
    def generate_testing_plan(self):
        """Generate a systematic testing plan for the new strategies."""
        print("\n" + "="*80)
        print("SYSTEMATIC TESTING PLAN")
        print("="*80)
        
        testing_phases = [
            {
                'phase': 'Phase 1: Bear Market Optimization',
                'period': '2021-11-01 to 2022-11-01',
                'strategies': ['bear-market-optimized-vp.yaml'],
                'success_criteria': [
                    'Positive returns (>0%)',
                    'Sharpe ratio >0.5',
                    'Max drawdown <20%',
                    'Win rate >35%'
                ]
            },
            {
                'phase': 'Phase 2: Win Rate Improvement',
                'period': '2022-11-01 to 2025-01-01',
                'strategies': ['multi-timeframe-vp-enhanced.yaml', 'improved-vp-fvg-simple.yaml'],
                'success_criteria': [
                    'Win rate >40%',
                    'Sharpe ratio >0.8',
                    'Profit factor >1.4',
                    'Lower trade frequency but higher quality'
                ]
            },
            {
                'phase': 'Phase 3: Overall Performance',
                'period': '2021-01-01 to 2025-01-01',
                'strategies': ['improved-volume-profile-v1.yaml'],
                'success_criteria': [
                    'Total return >150%',
                    'Sharpe ratio >1.0',
                    'Max drawdown <30%',
                    'Consistent performance across regimes'
                ]
            }
        ]
        
        for phase in testing_phases:
            print(f"\n📊 {phase['phase']}")
            print(f"Period: {phase['period']}")
            print(f"Strategies to test: {', '.join(phase['strategies'])}")
            print("Success Criteria:")
            for criteria in phase['success_criteria']:
                print(f"  ✓ {criteria}")
    
    def plot_performance_comparison(self):
        """Create visual comparison of current performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Current Volume Profile Strategy Performance Analysis', fontsize=16)
        
        periods = list(self.results.keys())
        
        # Returns comparison
        returns = [self.results[p]['total_return'] for p in periods]
        axes[0,0].bar(periods, returns, color=['red', 'orange', 'green'])
        axes[0,0].set_title('Total Returns by Period')
        axes[0,0].set_ylabel('Return (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Sharpe ratio comparison
        sharpe = [self.results[p]['sharpe_ratio'] for p in periods]
        axes[0,1].bar(periods, sharpe, color=['red', 'orange', 'green'])
        axes[0,1].set_title('Sharpe Ratio by Period')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='Target: 1.0+')
        axes[0,1].legend()
        
        # Win rate comparison
        win_rates = [self.results[p]['win_rate'] for p in periods]
        axes[1,0].bar(periods, win_rates, color=['red', 'orange', 'green'])
        axes[1,0].set_title('Win Rate by Period')
        axes[1,0].set_ylabel('Win Rate (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].axhline(y=50, color='blue', linestyle='--', alpha=0.7, label='Target: 50%+')
        axes[1,0].legend()
        
        # Max drawdown comparison
        drawdowns = [self.results[p]['max_drawdown'] for p in periods]
        axes[1,1].bar(periods, drawdowns, color=['red', 'orange', 'green'])
        axes[1,1].set_title('Maximum Drawdown by Period')
        axes[1,1].set_ylabel('Max Drawdown (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].axhline(y=25, color='blue', linestyle='--', alpha=0.7, label='Target: <25%')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('current_strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 Performance visualization saved as 'current_strategy_analysis.png'")

def main():
    """Run the complete strategy analysis."""
    analyzer = StrategyAnalyzer()
    
    # Run analysis
    df = analyzer.analyze_current_performance()
    analyzer.identify_weaknesses()
    analyzer.propose_improvements()
    analyzer.risk_management_recommendations()
    analyzer.generate_testing_plan()
    analyzer.plot_performance_comparison()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Test the bear market optimized strategy first:
   poetry run btc-backtest btc_research/config/bear-market-optimized-vp.yaml --plot

2. Compare with the simplified VP-FVG approach:
   poetry run btc-backtest btc_research/config/improved-vp-fvg-simple.yaml --plot

3. Evaluate the multi-timeframe enhanced strategy:
   poetry run btc-backtest btc_research/config/multi-timeframe-vp-enhanced.yaml --plot

4. Test the overall improved strategy:
   poetry run btc-backtest btc_research/config/improved-volume-profile-v1.yaml --plot

5. Compare all results and select the best-performing variant

6. Consider ensemble approach combining best elements from each strategy
    """)

if __name__ == "__main__":
    main()