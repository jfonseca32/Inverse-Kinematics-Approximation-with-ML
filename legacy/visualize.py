import pandas as pd
import matplotlib.pyplot as plt
import argparse

def visualize_coverage(csv_file: str):
    """
    Creates scatter plots of dataset coverage in XY, XZ, and YZ planes.
    """
    df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(df['x'], df['y'], s=1, alpha=0.3)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Plane')

    axes[1].scatter(df['x'], df['z'], s=1, alpha=0.3)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ Plane')

    axes[2].scatter(df['y'], df['z'], s=1, alpha=0.3)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ Plane')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize dataset coverage.')
    parser.add_argument('csv_file', type=str, help='CSV file to visualize (train or raw)')
    args = parser.parse_args()
    visualize_coverage(args.csv_file)

if __name__ == '__main__':
    main()
