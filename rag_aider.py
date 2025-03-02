#!/usr/bin/env python3

import sys
import os
import subprocess
import tempfile
import shutil

def main():
    """Create a simple wrapper that runs Aider with an environment variable."""
    # Get original args minus script name
    args = sys.argv[1:]
    
    # Create a temp file to tell Aider we want to use RAG
    marker_file = os.path.join(tempfile.gettempdir(), "aider_use_rag.txt")
    with open(marker_file, "w") as f:
        f.write("1")
    
    try:
        # Prepare environment variables
        env = os.environ.copy()
        env["AIDER_USE_RAG"] = "1"
        
        # Run regular Aider with our environment variable
        cmd = ["aider"] + args
        print(f"Starting Aider with RAG enhancement...")
        
        return subprocess.call(cmd, env=env)
    finally:
        # Clean up
        if os.path.exists(marker_file):
            os.remove(marker_file)

if __name__ == "__main__":
    sys.exit(main())