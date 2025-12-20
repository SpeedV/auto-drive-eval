#!/bin/bash
# White Agent Entrypoint

cd ../..
export ROLE=white
# Override PORT so it doesn't conflict with the Controller (8080)
# export PORT=8002
python main.py