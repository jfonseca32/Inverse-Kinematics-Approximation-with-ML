# Inverse Kinematics Approximation Using Machine Learning

# 1. Introduction / Background

Inverse Kinematics (IK) refers to finding specific angle configurations of a robotic arm’s joints that lead to a given end-effector position. Current approaches make use of numerical solvers, which can be slow and often provide only one solution when there are multiple possibilities. In this project, we aim to estimate IK mapping with supervised and unsupervised machine learning, reducing computational complexity, but still maintaining accuracy up to par with current methods.

Several recent works [1] explore the use of neural networks for IK, showing promising results. Unsupervised learning can also assist in understanding configuration space (the set of all possible positions the robot’s joints can take i.e. Joint 1 can rotate from 0° to 180°, Joint 2 from -90° to 90°, Joint 3 from 0° to 150°) and clustering feasible/optimal joint angle patterns for given tasks.

Our dataset will be synthetically generated using PyBullet or Mujoco physics simulator. By randomly sampling valid joint configurations of a 6-DOF robotic arm and recording corresponding end positions (x, y, z), we will build a dataset of 50,000+ (position, angles) pairs, using 70% for training, 15% for testing and 15% for validation (a conventional approach for ML projects [2]).

# 2. Problem Definition

The IK problem often faces multiple possible joint configurations for a single end position. Our objective is to develop a machine learning model that predicts both a feasible and accurate joint configuration given a target end location in 3D space. This approach enables real-time performance for control systems in robotics, replacing slower iterative solvers that are especially affected by creep and wearing of the arm in the long-run [3]. 

The IK concept can be extended into unsupervised learning for anomaly detection. A robotic manipulator is considered to be in a “displacement singularity” when its jacobian loses rank and one of the joint angles can be varied arbitrarily without causing a change in the end-effector position. Therefore, unsupervised learning can be used to identify regions of the dataset with an abnormally high density of inverse kinematic solutions, thus allowing us to identify singularities without training labels. 

Project outcomes:

a) Supervised learning model for inverse kinematic approximation/prediction. Given an end-effector position in task-space, we aim to approximate the corresponding angles in joint-space that produce the specified orientation at the end-effector.

b) Unsupervised learning model for displacement singularity detection. 

# 3. Methods

Preprocessing Methods:

- Normalization of joint angles and position vectors

- Outlier filtering using DBSCAN distance

- Dimensionality reduction via autoencoders (unsupervised) to make training faster


Supervised Learning Models:

- Multi-Layer Perceptron (MLP) using sklearn.neural_network.MLPRegressor

- Support Vector Regression using sklearn.svm.SVR

- Gradient Boosted Trees with sklearn.ensemble.GradientBoostingRegressor


Unsupervised Learning Methods:

- Clustering of joint angle configurations using K-Means to group similar poses together (like "elbow up" vs "elbow down")

- Singularity detection: Find nearest neighbors for a given point using  sklearn.neighbors.NearestNeighbors, PCA for dimensionality reduction with sklearn.decomposition.PCA, Isolation forest for anomaly detection using sklearn.ensemble.IsolationForest, DBSCAN to cluster singularity scores with sklearn.cluster.DBSCAN

Our implementation will primarily use Python with Scikit-learn, PyBullet, and TensorFlow/Keras for neural networks.

# 4. Potential Results and Discussion

We will evaluate model performance using:

- Mean Absolute Error (MAE) between predicted and actual joint angles

- Euclidean End-Effector Position Error after applying predicted angles via forward kinematics

- Computation Time for inference vs traditional IK solvers

- Simulations rendered by PyBullet or Mujoco framework.

We aim for sub-degree accuracy and sub-millisecond inference time. Additional goals include investigating multiple-solution handling and improving generalization across configurations. We will also discuss sustainability aspects by evaluating model energy usage during training and inference, and scalability to higher DOF systems.

# 5. References
[1] F. L. Tagliani, N. Pellegrini, and F. Aggogeri, “Machine learning sequential methodology for robot inverse kinematic modelling,” Applied Sciences, vol. 12, no. 19, p. 9417, Sep. 2022. doi: 10.3390/app12199417

[2] R. F. Reinhart, Z. Shareef, and J. J. Steil, “Hybrid analytical and data-driven modeling for feed-forward robot control,” Sensors, vol. 17, no. 2, p. 311, Feb. 2017. doi: 10.3390/s17020311

[3] M. N. Vu, F. Beck, M. Schwegel, C. Hartl-Nesic, A. Nguyen, and A. Kugi, “Machine learning-based framework for optimally solving the analytical inverse kinematics for redundant manipulators,” Mechatronics, vol. 89, p. 102970, 2023. doi: 10.1016/j.mechatronics.2023.102970
