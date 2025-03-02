import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_token_reductions(log_path):
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created directory: {log_dir}")

    if not os.path.exists(log_path):
        print(f"The file {log_path} does not exist. Creating a new file.")
        with open(log_path, 'w') as f:
            f.write("Timestamp,Reduction %\n")

    df = pd.read_csv(log_path)

    print("Token Reduction Summary:")
    print(f"Total Interactions: {len(df)}")
    print(f"Average Reduction: {df['Reduction %'].mean():.2f}%")
    print(f"Median Reduction: {df['Reduction %'].median():.2f}%")
    print(f"Max Reduction: {df['Reduction %'].max()}%")
    print(f"Min Reduction: {df['Reduction %'].min()}%")

    plt.figure(figsize=(10, 5))
    plt.plot(df['Timestamp'], df['Reduction %'])
    plt.title('Token Reduction Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Reduction Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/.aider/token_reduction_plot.png"))

if __name__ == "__main__":
    log_path = os.path.expanduser("~/.aider/rag_logs/token_reduction.csv")
    analyze_token_reductions(log_path)
