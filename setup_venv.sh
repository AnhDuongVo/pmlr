#!/bin/bash

# Delete virtual environment if it exists
if [ -d "pmlr_env" ]; then
    echo "Deleting existing virtual environment..."
    rm -rf pmlr_env
fi

# Create virtual environment
python3 -m venv pmlr_env

# Activate virtual environment
source pmlr_env/bin/activate

# Upgrade pip
pip3 install pip --upgrade

# Upgrade pip, wheel, and setuptools
pip3 install -U pip wheel setuptools

# Install concrete-ml version 1.4.1
pip3 install concrete-ml==1.4.1 matplotlib pandas

# Remove all caches
echo "Removing all caches..."
pip3 cache purge
