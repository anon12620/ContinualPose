#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # create virtual environment and install packages
  if [[ ! -d "venv" ]]; then
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -e .
  else
    source venv/bin/activate
  fi
  
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

# This runs your wrapped command
"$@"