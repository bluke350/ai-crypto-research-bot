import sys
import pytest
from contextlib import redirect_stdout

out_path = 'pytest_capture.txt'
with open(out_path, 'w', encoding='utf-8') as f:
    with redirect_stdout(f):
        # Use -vv for verbosity and stop on first failure to keep output manageable
        ret = pytest.main(['-vv', '--maxfail=1'])
print(f'pytest exit code: {ret}')
print(f'Wrote verbose pytest output to {out_path}')
