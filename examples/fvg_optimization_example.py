#!/usr/bin/env python3
"""
FVG Strategy Optimization Example

This example demonstrates how to use the new optimization framework
to optimize the Fair Value Gap (FVG) trading strategy parameters.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run FVG strategy optimization examples."""
    
    print("=== FVG Strategy Optimization Examples ===\n")
    
    # Configuration file path
    config_path = "btc_research/config/fvg-strategy.yaml"
    
    print("1. Basic Grid Search Optimization (Legacy compatible)")
    print("   Command:")
    print(f"   btc-optimise {config_path} --grid 'EMA_50.length=20,50,100;FVG_15m.min_gap_pips=5.0,10.0,15.0'")
    print()
    
    print("2. Bayesian Optimization with Walk-Forward Validation")
    print("   Command:")
    print(f"   btc-optimise {config_path} --method bayesian --validation walk-forward \\")
    print("     --grid 'EMA_50.length=20,200;FVG_15m.min_gap_pips=5.0,30.0;RSI_14.length=10,30' \\")
    print("     --max-iterations 50 --output fvg_bayesian_results.json")
    print()
    
    print("3. Genetic Algorithm with Robustness Testing")
    print("   Command:")
    print(f"   btc-optimise {config_path} --method genetic --validation time-series-split \\")
    print("     --robustness monte-carlo,bootstrap \\")
    print("     --grid 'EMA_50.length=20,200;FVG_15m.min_gap_pips=5.0,30.0;FVG_15m.max_lookback=100,1000' \\")
    print("     --max-iterations 100 --output fvg_genetic_results.xlsx")
    print()
    
    print("4. Comprehensive Optimization with Statistical Testing")
    print("   Command:")
    print(f"   btc-optimise {config_path} --method bayesian --validation walk-forward \\")
    print("     --robustness monte-carlo,parameter-sensitivity \\")
    print("     --statistics hypothesis,performance \\")
    print("     --grid 'EMA_50.length=20,200;FVG_15m.min_gap_pips=5.0,30.0;RSI_14.length=10,30' \\")
    print("     --max-iterations 100 --output fvg_comprehensive_results.html")
    print()
    
    print("5. Quick Random Search for Exploration")
    print("   Command:")
    print(f"   btc-optimise {config_path} --method random --validation purged-cv \\")
    print("     --grid 'EMA_50.length=20,200;FVG_15m.min_gap_pips=5.0,30.0' \\")
    print("     --max-iterations 30 --output fvg_random_results.csv")
    print()
    
    print("Key Parameters to Optimize:")
    print("- EMA_50.length: EMA period for trend detection (20-200)")
    print("- FVG_15m.min_gap_pips: Minimum gap size in USD (5.0-30.0)")
    print("- FVG_15m.max_lookback: Historical gaps to track (100-1000)")
    print("- RSI_14.length: RSI calculation period (10-30)")
    print()
    
    print("Expected Results:")
    print("- Bayesian optimization should find optimal parameters in 50-100 iterations")
    print("- Walk-forward validation prevents overfitting to historical data")
    print("- Robustness testing ensures strategy works across different market conditions")
    print("- Statistical testing provides confidence in parameter selection")
    print()
    
    print("To run an actual optimization, use one of the commands above.")
    print("Make sure you have the required data files in btc_research/data/binanceus/")

if __name__ == "__main__":
    main()