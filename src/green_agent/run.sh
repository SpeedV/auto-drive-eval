#!/bin/bash
# Green Agent Entrypoint

cd ../..
export ROLE=green
# Override PORT so it doesn't conflict with the Controller (8080)
# export PORT=8001
python main.py