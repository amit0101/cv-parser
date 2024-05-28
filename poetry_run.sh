#!/bin/bash

# Check if poetry is installed, if not, install it
if ! command -v poetry &> /dev/null
then
    echo "Poetry could not be found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3.10 -
else
    echo "Poetry is already installed."
fi

# Navigate to the project directory (assuming this script is in the project root)
cd "$(dirname "$0")"

# Configure poetry to create virtual environments in the project directory
poetry config virtualenvs.in-project true

# Install the dependencies and create the virtual environment
echo "Installing dependencies using Poetry..."
poetry install

# Make the script executable (if not already)
if [ ! -x "$(basename "$0")" ]; then
    echo "Making the script executable..."
    chmod +x "$(basename "$0")"
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
poetry shell

echo "Setup complete. The virtual environment is activated."
