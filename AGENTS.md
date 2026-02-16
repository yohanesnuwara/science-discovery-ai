AGENTS.md
# Science Discovery AI - Agent Guidelines

## Project Overview
Scientific discovery using agentic AI with sandboxing capabilities. Python project for scientific/ML workflows (not limited to PyTorch).

## Workspace

./science-discovery-ai

## Essential Instructions

1. **Understand user's intent** Read user form in the `./templates/FORM.md` and understand it carefully what the user wants.
2. **Put scripts inside folder `./.sandbox`** and run them from that folder
3. **Use `uv run python ...`** to run Python scripts
4. **If you need to install new libraries**, run `uv add <package>`
5. **You have access to GPU**, so everytime you run ML/neural network, run using GPU or CUDA setup
6. **If you create visualization or create any file**, put your plots or files inside `.output`

## Prohibitions

1. **Do not make a synthetic data** if you cannot parse the data. Tell truthfully you have issues with parsing data, or better solve it. 

## Running Scripts

Execute scripts from the project root using:

```bash
uv run python .sandbox/script_name.py
```

Scripts are run directly as Python files (no formal build system).

## Directory Structure

- **`.sandbox/`** - All Python scripts go here
- **`.output/`** - All output files (plots, CSVs, reports) go here
- **`.input/`** - All input data files go here

## Package Management

Install dependencies with:
```bash
uv add <package-name>
```

Sync existing dependencies:
```bash
uv sync
```

## GPU Usage

**Always use GPU/CUDA** for ML models and neural networks:
- XGBoost: Use `"device": "cuda"`, `"tree_method": "hist"`
- PyTorch: Enable CUDA by default
- CuPy/cuML: Use `.to('cuda')` for GPU arrays

## Visualization and Output

- Save plots to `.output/` directory
- Use matplotlib for scientific visualizations
- Save with high DPI (300) for publication quality
```python
plt.savefig('.output/plot_name.png', dpi=300, bbox_inches='tight')
```

## Key Dependencies

From [pyproject.toml](pyproject.toml):
- PyTorch: GPU-accelerated deep learning
- XGBoost: GPU-accelerated gradient boosting
- pandas/geopandas: Data and geospatial analysis
- matplotlib/cartopy: Scientific visualization
- scikit-learn: Classical ML algorithms

## Best Practices

- Always use GPU for ML/neural network tasks
- Organize output files in `.output/` directory
- Use meaningful variable names reflecting scientific context
- Include basic error handling for file I/O and API requests
- Follow existing code patterns in sample scripts