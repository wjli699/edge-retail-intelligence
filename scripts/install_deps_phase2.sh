#!/usr/bin/env bash
# Install Phase 2 dependencies: ZeroMQ C dev headers + Python binding.
set -euo pipefail

echo "[info] Installing libzmq-dev ..."
sudo apt install -y libzmq3-dev

echo "[info] Installing pyzmq ..."
pip3 install --user pyzmq

echo "[ok] Phase 2 dependencies installed."
echo "     C++ : $(pkg-config --modversion libzmq)"
echo "     Python: $(python3 -c 'import zmq; print(zmq.__version__)')"
