#!/bin/bash
# Setup script for Voice TTS MCP Server
# Run this script to create the conda environment and install dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="voice-tts-mcp"

echo "Setting up Voice TTS MCP Server..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Get the conda environment path
CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp env.example .env
    echo "Please edit .env to configure your TTS server URL"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To use this MCP server with Cursor, add the following to ~/.cursor/mcp.json:"
echo ""
cat << EOF
{
  "mcpServers": {
    "voice-tts": {
      "command": "${CONDA_ENV_PATH}/bin/python",
      "args": ["$SCRIPT_DIR/mcp_voice_tts.py"],
      "env": {
        "TTS_SERVER_URL": "http://10.111.79.180:8080",
        "DEFAULT_VOICE": "voice_reference.wav",
        "DEFAULT_LANGUAGE": "en"
      }
    }
  }
}
EOF
echo ""
echo "Then restart Cursor to load the MCP server."
