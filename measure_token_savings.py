#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def create_test_queries():
    """Creates a set of standardized test queries for measuring token reduction."""
    return [
        "Explain how the vector store works in this codebase",
        "What does the optimize_messages function do?",
        "How does the token reduction logging work?",
        "What's the relationship between patch_aider.py and rag_aider.py?",
        "How is the vector store data persisted?",
        "Can you explain the AiderPromptOptimizer class?",
        "What metrics are collected in this system?",
        "Show me how to use this RAG system",
        "What improvements could be made to this codebase?",
        "Compare the token reduction approach used here with alternatives"
    ]

def run_standard_test():
    """Runs a standard test with predefined queries and measures token reduction."""
    print("\n===== Running Standard Token Reduction Test =====\n")
    
    # Ensure RAG is enabled
    rag_aider_path = "rag_aider.py"
    if not os.path.exists(rag_aider_path):
        print(f"❌ {rag_aider_path} not found!")
        return False
    
    # Check if debug token logging script exists
    debug_script = "debug_token_logging.py"
    if not os.path.exists(debug_script):
        print(f"❌ {debug_script} not found!")
        return False
    
    # Clear previous token logs
    token_log_path = os.path.expanduser("~/.aider/rag_logs/token_reduction.csv")
    if os.path.exists(token_log_path):
        backup_path = f"{token_log_path}.{int(time.time())}.bak"
        os.rename(token_log_path, backup_path)
        print(f"✅ Backed up existing token log to {backup_path}")
    
    # Force patching OpenAI client
    if os.path.exists("force_patch.py"):
        print("Running force_patch.py to ensure OpenAI client is patched...")
        subprocess.run([sys.executable, "force_patch.py"], check=True)
    
    # Get test queries
    queries = create_test_queries()
    print(f"Prepared {len(queries)} standardized test queries")
    
    # Start Aider with RAG in the background
    print("\nStarting Aider with RAG enhancement... (press Ctrl+C when done)\n")
    print("===== TEST INSTRUCTIONS =====")
    print("1. When Aider starts, enter each query below one at a time")
    print("2. Wait for each response before entering the next query")
    print("3. After running all queries, type 'exit' to close Aider")
    print("4. This script will analyze the results automatically\n")
    
    print("===== TEST QUERIES =====")
    for i, query in enumerate(queries):
        print(f"{i+1}. {query}")
    
    print("\nStarting Aider in 5 seconds... (press Ctrl+C to cancel)")
    time.sleep(5)
    
    try:
        # Run Aider
        subprocess.run([sys.executable, rag_aider_path, "--model", "gpt-4o"], check=True)
    except KeyboardInterrupt:
        print("\nAider process terminated")
    except Exception as e:
        print(f"❌ Error running Aider: {e}")
    
    # Analyze results
    analyze_results()
    
    return True

def analyze_results():
    """Analyzes token reduction results and produces visualizations."""
    print("\n===== Analyzing Token Reduction Results =====\n")
    
    # Run debug token logging to check current state
    if os.path.exists("debug_token_logging.py"):
        subprocess.run([sys.executable, "debug_token_logging.py"], check=True)
    
    # Check if token reduction log exists
    token_log_path = os.path.expanduser("~/.aider/rag_logs/token_reduction.csv")
    if not os.path.exists(token_log_path):
        print(f"❌ Token log not found at {token_log_path}")
        print("No token reduction data to analyze. Try running the test again.")
        return False
    
    # Load token reduction data
    try:
        df = pd.read_csv(token_log_path)
        print(f"Loaded token reduction data with {len(df)} entries")
        
        if len(df) < 2:
            print("❌ Not enough data for meaningful analysis")
            print("Try running more queries to generate more data")
            return False
        
        # Basic statistics
        print("\n===== Token Reduction Statistics =====")
        avg_reduction = df['Reduction %'].mean()
        median_reduction = df['Reduction %'].median()
        max_reduction = df['Reduction %'].max()
        min_reduction = df['Reduction %'].min()
        
        print(f"Total API calls analyzed: {len(df)}")
        print(f"Average token reduction: {avg_reduction:.2f}%")
        print(f"Median token reduction: {median_reduction:.2f}%")
        print(f"Maximum token reduction: {max_reduction:.2f}%")
        print(f"Minimum token reduction: {min_reduction:.2f}%")
        
        # Calculate cost savings
        total_orig_tokens = df['OrigTokens'].sum() if 'OrigTokens' in df.columns else 0
        total_opt_tokens = df['OptTokens'].sum() if 'OptTokens' in df.columns else 0
        tokens_saved = total_orig_tokens - total_opt_tokens
        
        # Estimate cost savings (assuming $0.01 per 1K tokens for GPT-4)
        cost_per_1k = 0.01
        cost_savings = (tokens_saved / 1000) * cost_per_1k
        
        print(f"\nTotal original tokens: {total_orig_tokens}")
        print(f"Total optimized tokens: {total_opt_tokens}")
        print(f"Tokens saved: {tokens_saved} ({(tokens_saved/total_orig_tokens*100):.2f}% overall)")
        print(f"Estimated cost savings: ${cost_savings:.4f}")
        
        # Generate visualization
        output_dir = os.path.expanduser("~/.aider")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert timestamp to datetime if it's a string
        if 'Timestamp' in df.columns and isinstance(df['Timestamp'].iloc[0], str):
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Plot reduction percentage over time
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(df)), df['Reduction %'], marker='o')
        plt.title('Token Reduction Over Queries')
        plt.xlabel('Query Number')
        plt.ylabel('Reduction Percentage')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "token_reduction_plot.png")
        plt.savefig(plot_path)
        print(f"\n✅ Saved token reduction plot to {plot_path}")
        
        # If we have original and optimized token counts, create a comparison bar chart
        if 'OrigTokens' in df.columns and 'OptTokens' in df.columns:
            plt.figure(figsize=(14, 7))
            
            # Create positions for bars
            x = range(len(df))
            width = 0.35
            
            # Plot bars
            plt.bar([i - width/2 for i in x], df['OrigTokens'], width, label='Original Tokens')
            plt.bar([i + width/2 for i in x], df['OptTokens'], width, label='Optimized Tokens')
            
            # Add labels and title
            plt.xlabel('Query Number')
            plt.ylabel('Token Count')
            plt.title('Original vs. Optimized Tokens per Query')
            plt.xticks(x)
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Add percentage labels
            for i in range(len(df)):
                reduction = df['Reduction %'].iloc[i]
                plt.text(i, df['OrigTokens'].iloc[i] + 50, f"{reduction:.1f}%", 
                         ha='center', va='bottom', fontsize=8, rotation=90)
            
            plt.tight_layout()
            
            token_comparison_path = os.path.join(output_dir, "token_comparison_plot.png")
            plt.savefig(token_comparison_path)
            print(f"✅ Saved token comparison plot to {token_comparison_path}")
        
        print("\n===== Analysis Complete =====")
        
        return True
    except Exception as e:
        print(f"Error analyzing token reduction data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n===== Token Savings Measurement Tool =====\n")
    
    # Ask user what they want to do
    print("Options:")
    print("1. Run standard test with predefined queries")
    print("2. Analyze existing token reduction data")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        run_standard_test()
    elif choice == '2':
        analyze_results()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()
