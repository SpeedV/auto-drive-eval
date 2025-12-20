# #!/bin/bash
# # Install dependencies if not already present (optional for some setups)
# # pip install -r requirements.txt

# # Start the FastAPI server
# # python main.py --port $AGENT_PORT

# #!/bin/bash
# python main.py

#!/bin/bash

# 1. Determine the target folder based on the ROLE variable
# FIX: Use '=' instead of '==' for compatibility
if [ "$ROLE" = "white" ]; then
    TARGET_DIR="src/white_agent"
else
    # Default to green if ROLE is generic or missing
    TARGET_DIR="src/green_agent"
    ROLE="green"
fi

echo "🔵 Dispatcher: Mapping ROLE [$ROLE] to directory [$TARGET_DIR]..."

# 2. Check if the directory exists
if [ -d "$TARGET_DIR" ]; then
    cd "$TARGET_DIR"
else
    echo "❌ Error: Directory '$TARGET_DIR' not found!"
    echo "👉 Available directories in src/:"
    ls -d src/*/
    exit 1
fi

# 3. Execute the inner script
if [ -f "run.sh" ]; then
    chmod +x run.sh
    ./run.sh
else
    echo "❌ Error: No 'run.sh' found inside $TARGET_DIR!"
    exit 1
fi