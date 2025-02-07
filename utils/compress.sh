#!/bin/bash

# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

# Default values
goal_dir="/seed4d_xz/compressed" # Default goal directory
source_dir="/seed4d/data"       # Default source directory

# Usage function
usage() {
    echo "Usage: $0 --mode <dynamic|static> [--goal-dir <directory>] [--source-dir <directory>]"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            mode="$2"
            shift # Shift to next argument
            shift
            ;;
        --goal-dir)
            goal_dir="$2"
            shift
            shift
            ;;
        --source-dir)
            source_dir="$2"
            shift
            shift
            ;;
        *)
            usage
            ;;
    esac
done

# Validate mode
if [[ -z "$mode" || ( "$mode" != "static" && "$mode" != "dynamic" ) ]]; then
    echo "Error: Invalid or missing mode."
    usage
fi

# Validate goal directory
if [[ -z "$goal_dir" ]]; then
    echo "Error: Invalid or missing goal directory."
    usage
fi

# Validate source directory
if [[ ! -d "$source_dir" ]]; then
    echo "Error: Source directory '$source_dir' does not exist."
    exit 1
fi

# Define the base directory depending on the mode
base_dir="${source_dir}/${mode}"

#folders=("Town01" "Town02" "Town03" "Town04" "Town05" "Town06" "Town07" "Town10HD")
folders=("Town10HD")

# Ensure the goal directory exists
if [ ! -d "$goal_dir" ]; then
    echo "Creating goal directory: $goal_dir"
    mkdir -p "$goal_dir"
fi

# Loop through each folder
for folder in "${folders[@]}"; do
    folder_path="${base_dir}/${folder}"
    
    # Check if the folder exists
    if [ ! -d "$folder_path" ]; then
        echo "Folder $folder_path does not exist. Skipping..."
        continue
    fi

    # Create a tar.xz archive for each folder with verbose output
    echo "Compressing $folder_path..."
    
    # Use `pv` to show progress if it's available, otherwise fallback to `tar` without progress
    if command -v pv >/dev/null 2>&1; then
        tar -cf - "$folder_path" | pv -s $(du -sb "$folder_path" | awk '{print $1}') | xz -z - > "${goal_dir}/${mode}_${folder}.tar.xz"
    else
        tar -cJf "${goal_dir}/${mode}_${folder}.tar.xz" "$folder_path" --verbose
    fi

    echo "$folder_path has been compressed into ${goal_dir}/${mode}_${folder}.tar.xz"
done

echo "All folders have been processed."

# Example: bash utils/compress.sh --mode static
