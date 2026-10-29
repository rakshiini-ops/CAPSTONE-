import subprocess
import sys
from part_a import run as run_a
from part_b import run as run_b
from part_c import run as run_c
from part_d import run as run_d

if __name__ == '__main__':
    print('\n=== PART A ==='); run_a()
    print('\n=== VALIDATION ===')
    subprocess.run([sys.executable, 'src/validate_a.py'], check=True)
    print('\n=== PART B ==='); run_b()
    print('\n=== PART C ==='); run_c()
    print('\n=== PART D ==='); run_d()
