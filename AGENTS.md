AGENTS.md
# Science Discovery AI - Agent Guidelines

## Project Overview
Scientific discovery using agentic AI with GPU-accelerated ML workflows. Python project for geophysical data analysis, ML modeling, and scientific visualization.

## Build, Lint, and Test Commands

### Running Python Scripts
All scripts should be executed from the `.sandbox` directory using the project's virtual environment:

```bash
cd /workspace/science-discovery-ai
uv run python .sandbox/script_name.py
```

### No Build System
This project does not use a formal build system. Scripts are run directly as Python files.

### Linting
No linting configuration is currently defined. Follow code style conventions manually.

### Testing
No formal test suite exists. Test scripts in `.sandbox/test_*.py` and `test.py` provide basic examples but are not executed as part of CI/CD.

## Code Style Guidelines

### Imports
Organize imports in the following order:
1. Standard library imports (e.g., `os`, `sys`, `datetime`)
2. Third-party imports (e.g., `numpy`, `pandas`, `xgboost`)
3. Local imports (if any)

```python
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from my_local_module import MyClass
```

### File Structure and Organization
- Place all main scripts in `.sandbox/`
- Place all output files (plots, CSVs, reports) in `.output/`
- Place all input data files in `.input/`
- Module files should be in the project root or appropriate subdirectories

### Naming Conventions
- **Files**: use lowercase with hyphens or underscores (e.g., `plate_stress_analysis.py`, `xgboost_gpu.py`)
- **Classes**: PascalCase (e.g., `PlateStressAnalyzer`, `XGBClassifierGPU`)
- **Functions**: snake_case (e.g., `load_data`, `run_full_analysis`)
- **Variables**: snake_case (e.g., `earthquake_data`, `n_estimators`)
- **Constants**: UPPERCASE_SNAKE_CASE (e.g., `SHEAR_MODULUS`, `CORE_DENSITY`)
- **Private methods**: underscore prefix (e.g., `_load_data`, `__init__`)

### Docstrings
Include docstrings for functions and classes using triple quotes. Document parameters and return values.

```python
def calculate_seismic_moments(self):
    """
    Calculate Seismic Moment (Mo) for each earthquake.

    Parameters:
    -----------
    None (uses `self`)

    Returns:
    --------
    DataFrame
        Updated DataFrame with seismic moment calculations
    """
```

### Type Hints
Use type hints for function parameters and return values where appropriate, especially for public APIs:
```python
def main() -> None:
    """Main execution function."""
```

### Error Handling
- Use try/except blocks for external API calls and file I/O
- Print meaningful error messages
- Use built-in `raise_for_status()` for HTTP requests
- Use specific exception handling rather than bare except clauses
```python
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except Exception as e:
    print(f"Error: {e}")
```

### GPU Acceleration
Always use GPU/CUDA when training ML models or performing numerical computations:
- XGBoost: `"device": "cuda"`, `"tree_method": "hist"`
- PyTorch: Enable CUDA by default
- CuPy/cuML: Use `.to('cuda')` for GPU arrays

### Scientific Visualization
- Use `matplotlib` for plots and save to `.output/` directory
- Recommended style: `plt.style.use('seaborn-v0_8-whitegrid')`
- Save figures with high DPI (300) for publication-quality outputs
- Include informative titles, axis labels, and legends
```python
plt.savefig(f'{output_dir}/plot_name.png', dpi=300, bbox_inches='tight')
```

### Logging and Progress
Use informative console output with clear indicators:
- Progress phase indicators: `[Phase 1]`, `[Phase 2]`
- Success indicators: `✓`, `Saved:`
- Error indicators: `✗`, `Error:`
- Warning indicators: `⚠`
```python
print("\n[Phase 1] Loading data...")
```

### Data I/O
- Use `pandas` for CSV data handling
- Use `geopandas` for geospatial data
- Use `os.makedirs(exist_ok=True)` to create output directories
- Reference files with absolute paths for consistency

### Pandas Best Practices
- Use descriptive column names
- Round statistics and results appropriately
- Group operations with `groupby().agg()`
- Use `pd.to_datetime()` for time conversions
```python
df = df.groupby('column').agg({'magnitude': ['mean', 'std']}).round(4)
```

### NumPy Best Practices
- Use `np.set_printoptions()` for consistent numeric output
- Use vectorized operations instead of loops when possible
- Use `np.column_stack()` for coordinate arrays
- Use `np.maximum()` to avoid division by zero
```python
np.set_printoptions(precision=4, suppress=True)
```

## Project-Specific Conventions

### Code Organization
- Follow the existing pattern of having a main class with modular methods
- Each method should have a clear, single responsibility
- Group related functionality together

### Scientific Analysis Workflow
1. Load data
2. Preprocess and clean
3. Perform calculations
4. Generate visualizations
5. Save results and reports

### Environment Variables
No specific environment variable configuration is required. All paths are relative to project root using `/workspace/science-discovery-ai/`

## Continuous Integration

No CI/CD configuration is defined. This project focuses on local development and experimentation using GPU resources.

## Dependencies

Key dependencies (from pyproject.toml):
- PyTorch: GPU-accelerated deep learning
- XGBoost: GPU-accelerated gradient boosting
- cuML/cuPy: GPU-accelerated ML and numerical computing
- pandas/geopandas: Data and geospatial analysis
- matplotlib/cartopy: Scientific visualization
- scikit-learn: Classical ML algorithms

## Additional Notes

- Run experiments with GPU enabled for optimal performance
- Keep output files organized in the `.output/` directory
- Use meaningful variable names that reflect scientific context
- Include basic error handling for file operations and API requests
- Follow the existing code patterns in sample scripts