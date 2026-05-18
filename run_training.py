import sys, os, json, subprocess, time

os.chdir('/home/kronos/College/Flash-IDS')

script_start = time.time()

print("=" * 60)
print("FLASH Training Pipeline — Execution via nbconvert")
print("=" * 60)

# Build command using venv's jupyter
venv_python = '/home/kronos/College/Flash-IDS/.venv/bin/python3'
nb_path = '/home/kronos/College/Flash-IDS/Cadets.ipynb'
output_path = '/home/kronos/College/Flash-IDS/Cadets_executed.ipynb'

cmd = [
    venv_python, '-m', 'jupyter', 'nbconvert', '--to', 'notebook',
    '--execute', nb_path,
    '--output', output_path,
    '--ExecutePreprocessor.timeout=7200',  # 2 hour timeout
]

print(f"Executing: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:", result.stdout[:2000] if result.stdout else "(empty)")
print()
if result.stderr:
    print("STDERR:", result.stderr[:2000] if result.stderr else "(empty)")
print()
print(f"Return code: {result.returncode}")

elapsed = time.time() - script_start
print(f"Total elapsed: {elapsed:.1f}s")

if result.returncode == 0:
    print("\n✓ Training pipeline completed successfully!")
    
    # Find benchmark results
    for f in os.listdir('.'):
        if f.startswith('benchmark_results_') and f.endswith('.json'):
            with open(f) as bf:
                data = json.load(bf)
            print(f"\nBenchmark results from {f}:")
            print(json.dumps(data, indent=2))
            break
else:
    print("\n✗ Training pipeline failed")
    sys.exit(1)
