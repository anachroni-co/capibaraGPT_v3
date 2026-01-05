#!/bin/bash

# Script para iniciar el servidor backend con E2B
export E2B_API_KEY="e2b_d8df23b5de5214b7bfb4ebe227a308b61a2ae172"

cd /home/elect/capibara6/backend
python3 capibara6_integrated_server.py > capibara6_server.log 2>&1 &