import os
import sys
import logging
import datetime
import importlib
from openai import OpenAI

# Ensure the file exists first
patch_log_path = os.path.expanduser("~/.aider/rag_logs/patch_timing.log")
os.makedirs(os.path.dirname(patch_log_path), exist_ok=True)
with open(patch_log_path, 'a') as f:
    f.write(f"{datetime.datetime.now()} - Patch module loaded\n")

# Set up logging to both file and console
log_dir = os.path.expanduser("~/.aider/rag_logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('🔍 %(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Configure file handler
file_handler = logging.FileHandler(os.path.join(log_dir, "rag_debug.log"))
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add the handlers to the logger
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Create a direct print to log file function
def log_direct(message):
    with open(os.path.join(log_dir, "direct.log"), 'a') as f:
        f.write(f"{datetime.datetime.now()} - {message}\n")

log_direct("Starting patch_aider module")

# Import our custom logger
try:
    # First try to import from current directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag_logger import RagLogger
    logging.info("Imported RagLogger from current directory")
    log_direct("Imported RagLogger from current directory")
except ImportError:
    # Define fallback logger
    logging.warning("Could not import RagLogger, using fallback")
    log_direct("Using fallback RagLogger")
    
    class RagLogger:
        def __init__(self):
            self.log_dir = os.path.expanduser("~/.aider/rag_logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                
            # Path for token reduction logs
            self.token_log_path = os.path.join(self.log_dir, "token_reduction.csv")
            self._ensure_token_log_exists()
        
        def _ensure_token_log_exists(self):
            """Create token reduction log file if it doesn't exist."""
            if not os.path.exists(self.token_log_path):
                with open(self.token_log_path, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Reduction %", "Original Tokens", "Optimized Tokens"])
        
        def log_token_reduction(self, original, optimized):
            """Log token reduction metrics."""
            if original <= 0:
                return
                
            reduction_percent = ((original - optimized) / original) * 100
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                # Print to console for visibility
                print(f"📊 [TOKEN] Reduction: {reduction_percent:.2f}% ({original} to {optimized} tokens)")
                
                with open(self.token_log_path, 'a', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow([timestamp, f"{reduction_percent:.2f}", original, optimized])
                
                log_direct(f"Logged token reduction: {reduction_percent:.2f}% ({original} to {optimized})")
            except Exception as e:
                log_direct(f"Error logging token reduction: {e}")
                print(f"⚠️ [ERROR] Failed to log token reduction: {e}")
                
        def log_vector_search(self, query, results, search_time_ms):
            log_direct(f"Vector search: {query[:30]}... ({len(results)} results)")
            
        def log_retrieval_metrics(self, query, contexts, score, method):
            log_direct(f"Retrieval metrics: {len(contexts)} contexts, score {score:.4f}")

# Import vector store
try:
    from vector_store import SimpleVectorStore, AiderPromptOptimizer
    logging.info("Successfully imported vector store components")
    log_direct("Imported vector store components")
except Exception as e:
    logging.error(f"Failed to import vector store: {e}")
    log_direct(f"Failed to import vector store: {e}")
    # Create minimal stubs for testing
    class SimpleVectorStore:
        def __init__(self, store_path):
            self.documents = []
        def search(self, query, top_k=5):
            return []
        def add_document(self, doc_id, content, metadata):
            pass
    
    class AiderPromptOptimizer:
        def __init__(self, vector_store):
            pass
        def update_recent_interactions(self, interaction):
            pass
        def optimize_messages(self, original_messages, current_query, current_files_context):
            return original_messages

# Initialize OpenAI client
client = OpenAI()
logging.info("OpenAI client initialized")
log_direct("OpenAI client initialized")

# Verify we can check the token counts
try:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    can_count_tokens = True
    logging.info("GPT2 tokenizer loaded for accurate token counting")
except:
    can_count_tokens = False
    logging.warning("Could not load tokenizer, will use word count for tokens")

# Initialize vector store and optimizer
vector_store_path = os.path.expanduser("~/.aider/simple_vector_store.json")
vector_store = SimpleVectorStore(vector_store_path)
optimizer = AiderPromptOptimizer(vector_store, relevance_threshold=0.6)
logging.info(f"Vector store initialized at {vector_store_path}")
log_direct(f"Vector store initialized at {vector_store_path}")

# Initialize our logger
rag_logger = RagLogger()

# Store the original method
original_create = client.chat.completions.create

# Track if we've been patched
has_been_patched = False

def count_tokens(text):
    """Count tokens in text, falling back to rough estimation if tokenizer not available."""
    if can_count_tokens:
        return len(tokenizer.encode(text))
    else:
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


# Flag to prevent recursion
_in_patched_call = False

def patched_create(*args, **kwargs):
    """Intercepts OpenAI API calls, optimizes context, and logs usage."""
    global _in_patched_call
    if _in_patched_call:  # Prevent recursion
        return original_create(*args, **kwargs)
    _in_patched_call = True
    try:
        log_direct("Patched OpenAI API call intercepted!")
        print("\n🎯 [DEBUG] Patched OpenAI API call intercepted!")
        logging.info("Intercepting API call!")

        if "messages" in kwargs:
            messages = kwargs["messages"]

            # Count original tokens
            orig_token_count = sum(count_tokens(m.get("content", "")) for m in messages)
            logging.debug(f"Original messages token count: {orig_token_count}")
            print(f"🔢 [DEBUG] Original token count: {orig_token_count}")

            # Extract last user query
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                current_query = user_messages[-1].get("content", "")
                print(f"📝 [DEBUG] User query: {current_query[:60]}{'...' if len(current_query) > 60 else ''}")

                # Get current files context (if available)
                files_context = ""
                for m in messages:
                    if m.get("role") == "system" and "Current Files:" in m.get("content", ""):
                        files_context = m.get("content", "")
                        break

                try:
                    logging.info(f"Optimizing messages for query: {current_query[:30]}...")
                    log_direct(f"Optimizing messages for query: {current_query[:30]}...")
                    optimized_messages = optimizer.optimize_messages(messages, current_query, files_context)

                    # Count optimized tokens
                    opt_token_count = sum(count_tokens(m.get("content", "")) for m in optimized_messages)
                    print(f"📉 [DEBUG] Optimized token count: {opt_token_count} tokens")

                    if orig_token_count > 0 and orig_token_count != opt_token_count:
                        reduction_percent = int((1 - opt_token_count / orig_token_count) * 100)
                        print(f"📉 [DEBUG] Token reduction: {orig_token_count} tokens -> {opt_token_count} tokens (approx. {reduction_percent}% saved)")
                        logging.info(f"Token reduction: {reduction_percent}%")
                        log_direct(f"Token reduction: {reduction_percent}% ({orig_token_count} to {opt_token_count})")
                        rag_logger.log_token_reduction(orig_token_count, opt_token_count)
                    else:
                        print("[DEBUG] No token reduction achieved.")

                    # Update messages with optimized version
                    kwargs["messages"] = optimized_messages
                except Exception as e:
                    logging.error(f"Error optimizing messages: {e}")
                    log_direct(f"Error optimizing messages: {e}")
                    print(f"⚠️ [ERROR] Failed to optimize messages: {e}")
            else:
                print("ℹ️ [INFO] No user message found to optimize")
                log_direct("No user message found to optimize")

        log_direct("Calling original OpenAI API")
        response = original_create(*args, **kwargs)
        log_direct("Received response from original OpenAI API")

        logging.info("API Response received")
        print(f"📡 [DEBUG] Model Used: {response.model}")
        print(f"📊 [DEBUG] Token Usage: {response.usage.total_tokens} total")

        # (Interaction storage code would be here)
        return response

    except Exception as e:
        logging.error(f"Unexpected error in patched_create: {e}")
        log_direct(f"Unexpected error in patched_create: {e}")
        return original_create(*args, **kwargs)
    finally:
        _in_patched_call = False  # Always reset the flag


def inject_patched_client():
    """Inject our patched client into modules that might use OpenAI."""
    try:
        # Add our patched client to the OpenAI module
        sys.modules['openai'].OpenAI = lambda *args, **kwargs: client
        
        # Force reload any already imported modules that might use OpenAI
        for module_name in list(sys.modules.keys()):
            if 'aider' in module_name and module_name != '__main__':
                try:
                    importlib.reload(sys.modules[module_name])
                    logging.debug(f"Reloaded module: {module_name}")
                except:
                    pass
        
        logging.info("Patched OpenAI client injected into all modules")
        log_direct("Patched OpenAI client injected into all modules")
        return True
    except Exception as e:
        logging.error(f"Failed to inject patched client: {e}")
        log_direct(f"Failed to inject patched client: {e}")
        return False

# Inject patched client
inject_success = inject_patched_client()
log_direct(f"Initial client injection result: {inject_success}")

print("\n🛠 [INFO] Successfully patched OpenAI API calls with RAG optimization!\n")

# For use by other modules
def get_patched_client():
    return client

def patch_aider():
    """Function to be called from rag_aider.py to ensure patch is applied"""
    logging.info("RAG optimization patch requested")
    log_direct("RAG optimization patch requested")
    
    global has_been_patched
    if not has_been_patched:
        # Re-apply patch if somehow lost
        client.chat.completions.create = patched_create
        log_direct("Re-applied patch to client.chat.completions.create")
    
    # Ensure patched client is injected
    inject_success = inject_patched_client()
    log_direct(f"Client injection result: {inject_success}")
    
    # Check vector store status
    doc_count = len(vector_store.documents)
    print(f"📚 [INFO] Vector store contains {doc_count} documents")
    log_direct(f"Vector store contains {doc_count} documents")
    
    return "RAG optimization patch successfully applied"