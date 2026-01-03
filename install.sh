#!/bin/bash

uv init . -p 3.13
uv venv -p 3.13

source .venv/bin/activate

uv pip install math-verify pandas seaborn ipykernel ipywidgets