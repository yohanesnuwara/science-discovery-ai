# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific discovery using agentic AI. Python project targeting scientific/ML workflows with PyTorch.

## Package Management

Uses `uv` for dependency management. Python 3.10+.

```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package>

# Run Python scripts
uv run python main.py
uv run python test.py
```

## Environment Setup

The README contains instructions for setting up the development environment on Runpod with Ollama and Claude Code.