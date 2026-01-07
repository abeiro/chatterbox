#!/bin/bash

set -e  # Exit immediately if a command fails

BASE_DIR="/home/dwemer"
REPO_URL="https://github.com/Dwemer-Dynamics/chatterbox"
REPO_DIR="$BASE_DIR/chatterbox"
VENV_DIR="$REPO_DIR/venv"

echo "=== CHIM Chatterbox setup ==="

# Ensure base directory exists
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# Clone or update repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning Chatterbox repository..."
    git clone "$REPO_URL"
else
    echo "Repository already exists, pulling latest changes..."
    cd "$REPO_DIR"
    git pull
fi

cd "$REPO_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .
pip install uvicorn fastapi

# Hugging Face login
echo
echo "=== Hugging Face Authentication ==="
echo "You may be prompted to log in to Hugging Face."
echo "This is required to download certain models."
echo
hf auth login

# User notice before launching service
echo
echo "This will start CHIM XTTS to download the selected model"
echo "Wait for the message:"
echo "  'Uvicorn running on http://0.0.0.0:8020 (Press CTRL+C to quit)'"
echo "Then close this window."
echo
echo "Press ENTER to continue"
read

# Launch the service
python3 restapi.py
