#!/bin/bash

# Script to copy CSV files from NBR-Restrictor/output to dataset/ directory
# Run this before converting to JSON format

SOURCE_DIR="/home/arkastor/Development/NBR-Restrictor/output"
DEST_DIR="/home/arkastor/Development/A-Next-Basket-Recommendation-Reality-Check/dataset"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy CSV files from source to destination
echo "Copying CSV files from $SOURCE_DIR to $DEST_DIR..."

if [ -d "$SOURCE_DIR" ]; then
    cp "$SOURCE_DIR"/*.csv "$DEST_DIR/" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "Successfully copied CSV files:"
        ls -la "$DEST_DIR"/*.csv
    else
        echo "No CSV files found in $SOURCE_DIR or copy failed"
    fi
else
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

echo "Dataset copy completed."
