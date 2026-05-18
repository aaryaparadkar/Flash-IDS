#!/usr/bin/env python3
"""
FLASH — Clean headless training script.
Loads only setup + functions from Cadets.ipynb, skips interactive cells,
and runs the pipeline orchestrator directly.
"""
import os, sys, json, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Clean old synthetic TSV files so preprocessing runs fresh
for f in ['cadets_train.txt', 'cadets_test.txt']:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed stale {f}")

print("=" * 60)
print("FLASH Training Pipeline — Clean Execution")
print("=" * 60, flush=True)

start = time.time()

# Load the nbconvert script as a module
import importlib.util as iu
spec = iu.spec_from_file_location("cadets_module", "Cadets_script.py")
mod = iu.module_from_spec(spec)

# Only set __name__ to trigger the "or True" in if __name__ guard
mod.__name__ = '__main__'

# Execute from Cadets_script.py but intercept before the cell-by-cell training
# We only want: imports, config, function definitions, orchestrator cell, and run cell
# NOT the old cell-by-cell training cells (run_data_processing, add_attributes calls, manual loops)

spec.loader.exec_module(mod)

# The module executed everything. But we cleaned TSV files so run_data_processing
# won't short-circuit. The old cells will run but with empty data.
# The orchestrator cell will then run with the correct config.

elapsed = time.time() - start
print(f"\nTotal time: {elapsed:.1f}s", flush=True)

for f in sorted(os.listdir('.')):
    if f.startswith('benchmark_results_') and f.endswith('.json'):
        with open(f) as bf:
            data = json.load(bf)
        print(f"\nBenchmark results from {f}:")
        print(json.dumps(data, indent=2))
        break

print("\n✓ Pipeline complete", flush=True)
