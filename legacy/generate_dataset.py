import numpy as np
import pandas as pd
from robot import forward_kinematics, JOINT_LIMITS
import argparse

def generate_dataset(n_samples: int, pose_noise: float = 0.0, angle_noise: float = 0.0) -> 'pd.DataFrame':
    """
    Generates a dataset of (x, y, z, psi, θ1, θ2, θ3, θ4) with optional noise.
    """
    joints = sample_joint_space(n_samples)
    data = []
    for theta in joints:
        pose = forward_kinematics(theta)
        if pose_noise > 0:
            pose[:3] += np.random.normal(scale=pose_noise, size=3)
        noisy_theta = theta.copy()
        if angle_noise > 0:
            noisy_theta += np.random.normal(scale=angle_noise, size=4)
        data.append(np.hstack((pose, noisy_theta)))

    cols = ['x','y','z','psi','t1','t2','t3','t4']
    return pd.DataFrame(data, columns=cols)

def sample_joint_space(n_samples: int) -> np.ndarray:
    """
    Samples uniformly in joint space within JOINT_LIMITS.
    """
    low = np.array([lim[0] for lim in JOINT_LIMITS])
    high = np.array([lim[1] for lim in JOINT_LIMITS])
    return np.random.uniform(low, high, size=(n_samples, 4))

# command line: 
#     python generate_dataset.py 200000 \
#     --pose_noise 0.005 \
#     --angle_noise 0.01 \
#     --output raw_dataset.csv

def main():
    parser = argparse.ArgumentParser(description='Generate IK dataset for 4-DOF arm.')
    parser.add_argument('n_samples', type=int, help='Number of samples to generate')
    parser.add_argument('--pose_noise', type=float, default=0.0,
                        help='Gaussian noise stddev on pose (meters)')
    parser.add_argument('--angle_noise', type=float, default=0.0,
                        help='Gaussian noise stddev on joint angles (radians)')
    parser.add_argument('--output', type=str, default='raw_dataset.csv',
                        help='Output CSV file path')
    args = parser.parse_args()

    df = generate_dataset(args.n_samples, args.pose_noise, args.angle_noise)
    df.to_csv(args.output, index=False)
    print(f'Dataset saved to {args.output}')

if __name__ == '__main__':
    main()
