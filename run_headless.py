#!/usr/bin/env python3
"""
FLASH headless execution runner.
Converts notebook → patched script → executes pipeline.
"""
import sys, os, json, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('Cadets_script.py') as f:
    script = f.read()

# Remove IPython magics
script = script.replace("get_ipython().run_line_magic('matplotlib', 'inline')", "")
script = script.replace("get_ipython().run_line_magic('matplotlib', 'inline')", "")

# Also remove the from IPython import if present (safe to just comment)
# Remove from IPython import warnings if any
script = script.replace("from IPython import get_ipython", "# from IPython import get_ipython")

print("=" * 60)
print("FLASH Training Pipeline — Headless Execution")
print("=" * 60, flush=True)

start = time.time()

# Use a fresh namespace to avoid conflicts
ns = {'__name__': '__main__', '__file__': 'Cadets_script.py'}
ns['__builtins__'] = __builtins__

try:
    exec(compile(script, 'Cadets_script.py', 'exec'), ns)
except SystemExit:
    pass
except Exception as e:
    print(f"\n✗ Pipeline failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

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
