#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <local_path> <username> <remote_path>"
  echo "local_path can be a file or a directory (relative to this script)."
  echo "remote_path is the target directory on the remote server (relative to user's home)."
  exit 1
fi

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Assign command-line arguments to variables
LOCAL_PATH="$SCRIPT_DIR/$1"  # Resolve local_path relative to the script
USERNAME=$2
REMOTE_PATH=$3
REMOTE_HOST="132.68.39.159"  # Fixed IP address

# Check if the local path exists
if [ ! -e "$LOCAL_PATH" ]; then
  echo "Error: Path '$LOCAL_PATH' does not exist."
  exit 1
fi

# Determine if the path is a file or directory
if [ -d "$LOCAL_PATH" ]; then
  TYPE="directory"
elif [ -f "$LOCAL_PATH" ]; then
  TYPE="file"
else
  echo "Error: '$LOCAL_PATH' is neither a file nor a directory."
  exit 1
fi

# Copy the file or directory to the remote server
echo "Copying $TYPE '$LOCAL_PATH' to '$USERNAME@$REMOTE_HOST:~/$REMOTE_PATH'..."
scp -r "$LOCAL_PATH" "$USERNAME@$REMOTE_HOST:~/$REMOTE_PATH"

# Check if the scp command was successful
if [ $? -eq 0 ]; then
  echo "$TYPE '$LOCAL_PATH' successfully copied to '$USERNAME@$REMOTE_HOST:~/$REMOTE_PATH'."
else
  echo "Error: Failed to copy the $TYPE."
  exit 1
fi
