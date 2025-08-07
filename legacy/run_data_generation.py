from generate_dataset import generate_dataset

# arguments: n_samples, pose_noise (m), angle_noise (rad)
df = generate_dataset(200_000, pose_noise=0.005, angle_noise=0.01)

# save to disk
df.to_csv('raw_dataset.csv', index=False)
print("Saved to raw_dataset.csv")
