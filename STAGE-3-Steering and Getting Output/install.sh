#!/bin/bash
pip install -U transformers torch accelerate && \
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121