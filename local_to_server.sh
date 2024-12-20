#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <local_directory> <username>"
  exit 1
fi

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Assign command-line arguments to variables
LOCAL_DIRECTORY="$SCRIPT_DIR/$1"
USERNAME=$2
REMOTE_HOST="132.68.39.159"
REMOTE_PATH="/home/$USERNAME/"

# Check if the local directory exists
if [ ! -d "$LOCAL_DIRECTORY" ]; then
  echo "Error: Directory '$LOCAL_DIRECTORY' does not exist."
  exit 1
fi

# Copy the directory to the remote server
echo "Copying directory '$LOCAL_DIRECTORY' to '$USERNAME@$REMOTE_HOST:$REMOTE_PATH'..."
scp -r "$LOCAL_DIRECTORY" "$USERNAME@$REMOTE_HOST:$REMOTE_PATH"

# Check if the scp command was successful
if [ $? -eq 0 ]; then
  echo "Directory '$LOCAL_DIRECTORY' successfully copied to '$USERNAME@$REMOTE_HOST:$REMOTE_PATH'."
else
  echo "Error: Failed to copy the directory."
  exit 1
fi
