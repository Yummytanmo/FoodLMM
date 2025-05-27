#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create logs directory if it doesn't exist
mkdir -p $DIR/logs

# Start the server
echo "Starting FoodLMM server on port 8001..."
python $DIR/server.py 2>&1 | tee $DIR/logs/server.log 