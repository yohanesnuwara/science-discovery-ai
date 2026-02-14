# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific discovery using agentic AI with sandboxing capabilities. Python project for scientific/ML workflows (not limited to PyTorch).

## Instructions

1. Understand user's intent and write a Python script to answer the intent. 
2. Put scripts inside folder `./.sandbox` and run them from that folder
3. Use `uv run python ...` to run Python scripts
4. If you need to install new libraries, run `uv add <package>`
5. You have access to GPU, so everytime you run ML/neural network, run using GPU or CUDA setup
6. If you create visualization or create any file, put your plots or files inside `.output`

## Environment Setup

The README contains instructions for setting up the development environment on Runpod with Ollama and Claude Code.