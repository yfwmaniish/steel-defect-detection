#!/bin/bash

# Steel Defect Detection - Edge Deployment Script

# Set variables
CONFIG_DIR="deployment/config"
CONFIG_FILE="${CONFIG_DIR}/edge_config.yaml"
MODEL_DIR="models"
MODEL_FILE="${MODEL_DIR}/steel_defect_advanced.pt"
DB_DIR="data"
DB_FILE="${DB_DIR}/inspections.db"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/edge_detector.log"

# Create necessary directories
mkdir -p ${CONFIG_DIR}
mkdir -p ${MODEL_DIR}
mkdir -p ${DB_DIR}
mkdir -p ${LOG_DIR}

# Copy configuration file
cp -n ${CONFIG_FILE} /etc/steel_defect_detection/edge_config.yaml

# Install system packages
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-opencv

# Install Python dependencies
pip3 install -r requirements.txt

# Check for model file
if [ ! -f ${MODEL_FILE} ]; then
    echo "ERROR: Model file ${MODEL_FILE} does not exist. Please provide the trained model file."
    exit 1
fi

# Setup database
if [ ! -f ${DB_FILE} ]; then
    echo "Initializing database..."
    sqlite3 ${DB_FILE} < scripts/setup_database.sql
fi

# Update log files
if [ ! -f ${LOG_FILE} ]; then
    touch ${LOG_FILE}
fi

# Set permissions
chmod 755 deployment/edge_detector.py
chmod 644 ${CONFIG_FILE}
chmod 644 ${DB_FILE}
chmod 644 ${LOG_FILE}

# Start the edge detection system
python3 deployment/edge_detector.py &

echo "Deployment completed successfully! Edge detection system is now running."

