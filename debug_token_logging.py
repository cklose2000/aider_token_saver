#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd

def check_token_logs():
    """Check if token reduction logs exist and verify their contents."""
    token_log_path = os.path.expanduser("~/.aider/rag_logs/token_reduction.csv")
    
    print(f"\n==== Token Reduction Log Analysis ====")
    if not os.path.exists(token_log_path):
        print(f"Token log file does not exist: {token_log_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(token_log_path)
    print(f"Token log file size: {file_size} bytes")
    
    # Read log file
    try:
        df = pd.read_csv(token_log_path)
        print(f"Successfully read token log file")
        print(f"Number of token reduction entries: {len(df)}")
        
        if len(df) > 0:
            print(f"Average reduction: {df['Reduction %'].mean():.2f}%")
            print(f"Most recent reductions:")
            print(df.tail(5))
        else:
            print("No token reduction entries found")
    except Exception as e:
        print(f"Error reading token log file: {e}")
        # Show file contents
        try:
            with open(token_log_path, 'r') as f:
                content = f.read()
                print(f"File contents: {content[:500]}...")
        except Exception as e2:
            print(f"❌ Could not read file contents: {e2}")
    
    return True

def check_vector_store():
    """Check if vector store exists and verify its contents."""
    vector_store_path = os.path.expanduser("~/.aider/simple_vector_store.json")
    
    print(f"\n==== Vector Store Analysis ====")
    if not os.path.exists(vector_store_path):
        print(f"❌ Vector store file does not exist: {vector_store_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(vector_store_path)
    print(f"Vector store file size: {file_size} bytes")
    
    # Read vector store
    try:
        with open(vector_store_path, 'r') as f:
            data = json.load(f)
            doc_count = len(data.get('documents', []))
            word_count = len(data.get('word_to_idx', {}))
            
            print(f"Successfully read vector store file")
            print(f"Number of documents: {doc_count}")
            print(f"Vocabulary size: {word_count}")
            
            # Count document types
            doc_types = {}
            for doc in data.get('documents', []):
                doc_type = doc.get('metadata', {}).get('type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            print(f"Document types:")
            for doc_type, count in doc_types.items():
                print(f"  - {doc_type}: {count}")
            
            # Show sample documents
            print(f"\nSample documents:")
            for i, doc in enumerate(data.get('documents', [])[:3]):
                doc_id = doc.get('doc_id', 'unknown')
                content_preview = doc.get('content', '')[:100].replace('\n', ' ')
                print(f"  {i+1}. {doc_id}: {content_preview}...")
    except Exception as e:
        print(f"Error reading vector store file: {e}")
    
    return True

def check_debug_logs():
    """Check if debug logs exist and verify their contents."""
    debug_log_path = os.path.expanduser("~/.aider/rag_logs/rag_debug.log")
    
    print(f"\n==== Debug Log Analysis ====")
    if not os.path.exists(debug_log_path):
        print(f"Debug log file does not exist: {debug_log_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(debug_log_path)
    print(f"Debug log file size: {file_size} bytes")
    
    # Count interesting log entries
    try:
        with open(debug_log_path, 'r') as f:
            content = f.read()
            intercept_count = content.count("Intercepting API call")
            optimize_count = content.count("Optimizing messages")
            token_count = content.count("Token reduction")
            error_count = content.count("ERROR")
            
            print(f"API intercepts: {intercept_count}")
            print(f"Message optimizations: {optimize_count}")
            print(f"Token reductions logged: {token_count}")
            print(f"Errors: {error_count}")
            
            # Show recent errors if any
            if error_count > 0:
                print(f"\nRecent errors:")
                error_lines = [line for line in content.split('\n') if "ERROR" in line]
                for line in error_lines[-3:]:
                    print(f"  {line}")
    except Exception as e:
        print(f"Error reading debug log file: {e}")
    
    return True

def inject_test_data():
    """Inject test data into token reduction log for visualization."""
    token_log_path = os.path.expanduser("~/.aider/rag_logs/token_reduction.csv")
    
    print(f"\n==== Injecting Test Data ====")
    try:
        # Ensure directory exists
        log_dir = os.path.dirname(token_log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created directory: {log_dir}")
        
        # Create or append to file
        file_exists = os.path.exists(token_log_path)
        mode = 'a' if file_exists else 'w'
        
        with open(token_log_path, mode) as f:
            if not file_exists:
                f.write("Timestamp,Reduction %\n")
            
            # Add test data for visualization
            import datetime
            base_date = datetime.datetime.now()
            for i in range(10):
                date = base_date - datetime.timedelta(hours=i)
                timestamp = date.strftime("%Y-%m-%d %H:%M:%S")
                reduction = 5 + i * 3  # Increases from 5% to 32%
                f.write(f"{timestamp},{reduction}\n")
        
        print(f"Added 10 test data points to token log for visualization")
    except Exception as e:
        print(f"Error injecting test data: {e}")

if __name__ == "__main__":
    print("\n===== RAG System Diagnostics =====\n")
    
    # Check if analysis only or with test data injection
    inject_test = '--inject-test' in sys.argv
    
    check_token_logs()
    check_vector_store()
    check_debug_logs()
    
    if inject_test:
        inject_test_data()
        print("\n==== After Test Data Injection ====")
        check_token_logs()
    
    print("\n===== Diagnostics Complete =====\n")
