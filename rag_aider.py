#!/usr/bin/env python3

import sys
import os
import subprocess
import tempfile
import logging
# Force patch application
import patch_aider
if hasattr(patch_aider, 'patch_aider'):
    result = patch_aider.patch_aider()
    print(f"Patch applied: {result}")

# Set up logging
log_dir = os.path.expanduser("~/.aider/rag_logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=os.path.join(log_dir, "rag_debug.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Create a wrapper that runs Aider with RAG enhancement."""
    # Get original args minus script name
    args = sys.argv[1:]
    
    logging.debug("Starting main function execution.")
    import patch_aider
    if hasattr(patch_aider, 'patch_aider'):
        result = patch_aider.patch_aider()
        print(f"Patch applied: {result}")
    # Force patch application
    import patch_aider
    if hasattr(patch_aider, 'patch_aider'):
        result = patch_aider.patch_aider()
        print(f"Patch applied: {result}")
    logging.info(f"Arguments: {args}")
    
    # Create a temp file to tell Aider we want to use RAG
    marker_file = os.path.join(tempfile.gettempdir(), "aider_use_rag.txt")
    with open(marker_file, "w") as f:
        f.write("1")
    logging.info(f"Created marker file: {marker_file}")
    
    try:
        # Prepare environment variables
        env = os.environ.copy()
        env["AIDER_USE_RAG"] = "1"
        logging.info("Set AIDER_USE_RAG=1 in environment")
        
        # Ensure our path is in PYTHONPATH
        current_dir = os.path.dirname(os.path.abspath(__file__))
        python_path = env.get("PYTHONPATH", "")
        if current_dir not in python_path:
            env["PYTHONPATH"] = f"{current_dir}:{python_path}" if python_path else current_dir
        logging.info(f"Set PYTHONPATH to include: {current_dir}")
        
        # Log which Python is being used
        python_path = subprocess.check_output(["where", "python"]).decode().strip()
        logging.info(f"Using Python from: {python_path}")
        
        # Run regular Aider with our environment variable
        cmd = ["aider"] + args
        logging.info(f"Starting Aider with command: {cmd}")
        print(f"Starting Aider with RAG enhancement (debug mode)...")
        
        # Import vector_store and patch_aider to verify they're available
        try:
            logging.info("Attempting to import modules...")
            from vector_store import SimpleVectorStore
            logging.info("Successfully imported SimpleVectorStore")
            import patch_aider
            logging.info("Successfully imported patch_aider")
        except Exception as e:
            logging.error(f"Error importing modules: {e}")
            print(f"Error initializing RAG system: {e}")
        
        # Check if vector store exists before running
        vector_store_path = os.path.expanduser("~/.aider/simple_vector_store.json")
        if os.path.exists(vector_store_path):
            logging.info(f"Vector store exists before running: {vector_store_path}")
            logging.info(f"Vector store size: {os.path.getsize(vector_store_path)} bytes")
        else:
            logging.info(f"Vector store does not exist before running")
        
        # Run Aider
        logging.debug("Executing Aider subprocess call...")
        result = subprocess.call(cmd, env=env)
        logging.info(f"Aider exited with code: {result}")
        
        # Check if vector store was created
        if os.path.exists(vector_store_path):
            logging.info(f"Vector store exists after running: {vector_store_path}")
            logging.info(f"Vector store size: {os.path.getsize(vector_store_path)} bytes")
        else:
            logging.info(f"Vector store still does not exist after running")
        
        return result
        
    except Exception as e:
        logging.error(f"Error running Aider: {e}")
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up
        if os.path.exists(marker_file):
            os.remove(marker_file)
            logging.info(f"Removed marker file: {marker_file}")
        logging.info("=== RAG-enhanced Aider session ended ===")

if __name__ == "__main__":
    sys.exit(main())
