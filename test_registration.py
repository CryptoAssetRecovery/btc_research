#!/usr/bin/env python3

"""Test script to verify indicator registration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from btc_research.core.registry import get_registry
    from btc_research.indicators import RiskManagement  # This should trigger registration
    
    registry = get_registry()
    print("Available indicators in registry:")
    for name in sorted(registry.keys()):
        print(f"  - {name}")
    
    if 'RiskManagement' in registry:
        print("\n✅ RiskManagement indicator is properly registered!")
        print("The backtest should now work.")
    else:
        print("\n❌ RiskManagement indicator is NOT registered.")
        print("There may be an issue with the indicator implementation.")
        
except Exception as e:
    print(f"Error during registration test: {e}")
    import traceback
    traceback.print_exc()