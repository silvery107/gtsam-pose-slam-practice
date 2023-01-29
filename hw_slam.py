import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gtsam

# Path to data folder
DATA_PATH = "data/"
FIGURE_PATH = "figures/"
DTYPE = np.float64

INTEL_2D = os.path.join(DATA_PATH, "input_INTEL_g2o.g2o")
GARAGE_3D = os.path.join(DATA_PATH, "parking-garage.g2o")

def load_g2o(filename):
    poses = []
    edges = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            temp = line.split()
            if temp[0][0] == "V":
                poses.append(temp[1:])
            elif temp[0][0] == "E":
                edges.append(temp[1:])
            else:
                raise NotImplementedError()
                
    data = {}
    data["poses"] = np.array(poses, dtype=DTYPE)
    data["edges"] = np.array(edges, dtype=DTYPE)
    
    return data

def construct_info_mat(info_v):
    info_m = None
    if len(info_v) == 6:
        info_m = np.zeros((3, 3), dtype=DTYPE)
        count = 0
        for i in range(3):
            for j in range(i, 3):
                info_m[i, j] = info_v[count]
                count += 1
  
    elif len(info_v) == 21:
        info_m = np.zeros((6, 6), dtype=DTYPE)
        count = 0
        for i in range(6):
            for j in range(i, 6):
                info_m[i, j] = info_v[count]
                count += 1
    else:
        raise NotImplementedError()
    
    info_m = info_m + info_m.T - np.diag(info_m.diagonal())
    return info_m

def plot_traj_2d(before, after, filename="test2d.png", axis_idx=[0, 1], show=False):
    axis_label = ["X axis", "Y axis", "Z axis"]
    assert before.shape[1] == after.shape[1] == 2
    plt.figure()
    plt.plot(before[:, 0], before[:, 1], 'b', label="Initial Trajectory")
    plt.plot(after[:, 0], after[:, 1], 'r', label="Optimized Trajectory")
    plt.xlabel(axis_label[axis_idx[0]])
    plt.ylabel(axis_label[axis_idx[1]])
    plt.grid()
    plt.legend()
    plt.axis("equal")
    plt.title(filename[:-4].replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, filename))
    if show:
        plt.show()

def plot_traj_3d(before, after, filename="test3d.png", show=False):
    assert before.shape[1] == after.shape[1] == 3
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(before[:, 0], before[:, 1], before[:, 2], 'b', label="Initial Trajectory")
    ax.plot(after[:, 0], after[:, 1], after[:, 2], 'r', label="Optimized Trajectory")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    # ax.set_box_aspect([1, 1, 1])
    ax.set_box_aspect((np.ptp(after[:, 0]), np.ptp(after[:, 1]), np.ptp(after[:, 2])))  # aspect ratio is 1:1:1 in data space

    plt.grid()
    plt.legend()
    plt.title(filename[:-4].replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, filename))
    if show:
        plt.show()

def solve_pose_slam_2d_batch(data, data_path):
    is3D = False
    graph, initial = gtsam.readG2o(data_path, is3D)

    # Add prior on the pose having index (key) = 0
    print("Adding prior to g2o file ")
    priorModel = gtsam.noiseModel.Diagonal.Variances(gtsam.Point3(1e-6, 1e-6, 1e-8))
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(), priorModel))

    params = gtsam.GaussNewtonParams()
    # The printing verbosity during optimization (default SILENT)
    params.setVerbosity("Termination") # this will show info about stopping conds
    # The maximum iterations to stop iterating (default 100)
    params.setMaxIterations(100)
    # The maximum relative error decrease to stop iterating (default 1e-5)
    params.setRelativeErrorTol(1e-5)

    # Create the optimizer ...
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
    # ... and optimize
    result = optimizer.optimize()

    print("Optimization complete")
    print("initial error = ", graph.error(initial))
    print("final error = ", graph.error(result))

    resultPoses = gtsam.utilities.extractPose2(result)
    initialPoses = data["poses"]
    
    plot_traj_2d(initialPoses[:, 1:3], resultPoses[:, :2], "solve_pose_slam_2d_batch.png")

def solve_pose_slam_2d_incremental(data):
    # Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
    # update calls are required to perform the relinearization.
    params = gtsam.ISAM2Params()
    # Only relinearize variables whose linear delta magnitude is greater than this threshold
    params.setRelinearizeThreshold(0.1)
    # Only relinearize any variables every relinearizeSkip calls to ISAM2::update
    params.relinearizeSkip = 10
    isam = gtsam.ISAM2(params)

    # Define the prior factor to the factor graph
    priorModel = gtsam.noiseModel.Diagonal.Variances(gtsam.Point3(1e-6, 1e-6, 1e-8))

    # Data initialize
    odometry_measurements = data["edges"]
    poses = data["poses"]

    # Initialize the current estimate which is used during the incremental inference loop.
    result = None
    for pose in tqdm(poses, desc="Processing"):
        # Create a Nonlinear factor graph as well as the data structure to hold state estimates.
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        id_p = int(pose[0])
        if id_p == 0:
            id_p, x, y, theta = pose
            id_p = id_p.astype(np.int32)
            
            graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(x, y, theta), priorModel))
            initial_estimate.insert(id_p, gtsam.Pose2(x, y, theta))
        else:
            prev_pose = result.atPose2(id_p - 1)
            initial_estimate.insert(id_p, prev_pose)
            for edge in odometry_measurements:
                if int(edge[1]) == id_p:
                    id_e1, id_e2, dx, dy, dtheta, *info = edge
                    id_e1 = id_e1.astype(np.int32)
                    id_e2 = id_e2.astype(np.int32)
                    
                    info_m = construct_info_mat(info)
                    noise_model = gtsam.noiseModel.Gaussian.Information(info_m)
                    graph.add(gtsam.BetweenFactorPose2(id_e1, id_e2 , gtsam.Pose2(dx, dy, dtheta), noise_model))
        
        # Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()

    resultPoses = gtsam.utilities.extractPose2(result)
    initialPoses = data["poses"]

    plot_traj_2d(initialPoses[:, 1:3], resultPoses[:, :2], "solve_pose_slam_2d_incremental.png")

def solve_pose_slam_3d_batch(data, data_path):
    is3D = True
    graph, initial = gtsam.readG2o(data_path, is3D)

    # Add Prior on the first key
    print("Adding prior to g2o file ")
    priorModel = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4], dtype=DTYPE))
    firstKey = initial.keys()[0]
    graph.add(gtsam.PriorFactorPose3(firstKey, gtsam.Pose3(), priorModel))

    params = gtsam.GaussNewtonParams()
    # The printing verbosity during optimization (default SILENT)
    params.setVerbosity("Termination") # this will show info about stopping conds
    # The maximum iterations to stop iterating (default 100)
    params.setMaxIterations(100)
    # The maximum relative error decrease to stop iterating (default 1e-5)
    params.setRelativeErrorTol(1e-5)

    # Create the optimizer ...
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
    # ... and optimize
    result = optimizer.optimize()

    print("Optimization complete")
    print("initial error = ", graph.error(initial))
    print("final error = ", graph.error(result))

    resultPoses = gtsam.utilities.extractPose3(result)
    initialPoses = data["poses"]
    
    plot_traj_3d(initialPoses[:, 1:4], resultPoses[:, -3:], "solve_pose_slam_3d_batch.png")
    plot_traj_2d(initialPoses[:, 1:3], resultPoses[:, -3:-1], "solve_pose_slam_3d_batch_x_y.png")
    plot_traj_2d(initialPoses[:, 2:4], resultPoses[:, -2:], "solve_pose_slam_3d_batch_y_z.png", [1, 2])

def solve_pose_slam_3d_incremental(data):
    # Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
    # update calls are required to perform the relinearization.
    params = gtsam.ISAM2Params()
    # Only relinearize variables whose linear delta magnitude is greater than this threshold
    params.setRelinearizeThreshold(0.1)
    # Only relinearize any variables every relinearizeSkip calls to ISAM2::update
    params.relinearizeSkip = 10
    isam = gtsam.ISAM2(params)

    # Define the prior factor to the factor graph
    priorModel = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4], dtype=DTYPE))
    # Data initialize
    odometry_measurements = data["edges"]
    poses = data["poses"]

    # Initialize the current estimate which is used during the incremental inference loop.
    result = None
    for pose in tqdm(poses, desc="Processing"):
        # Create a Nonlinear factor graph as well as the data structure to hold state estimates.
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        id_p = int(pose[0])
        if id_p == 0:
            id_p, x, y, z, *quat_v = pose
            id_p = id_p.astype(np.int32)

            rot_mat = gtsam.Rot3(quat_v[-1], *quat_v[:3])
            t_vec = gtsam.Point3(x, y, z)
            graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(rot_mat, t_vec), priorModel))
            initial_estimate.insert(id_p, gtsam.Pose3(rot_mat, t_vec))
        else:
            prev_pose = result.atPose3(id_p - 1)
            initial_estimate.insert(id_p, prev_pose)
            for edge in odometry_measurements:
                if int(edge[1]) == id_p:
                    id_e1, id_e2, dx, dy, dz, qx, qy, qz, qw, *info = edge
                    id_e1 = id_e1.astype(np.int32)
                    id_e2 = id_e2.astype(np.int32)
                    
                    rot_mat = gtsam.Rot3(qw, qx, qy, qz)
                    t_vec = gtsam.Point3(dx, dy, dz)
                    info_m = construct_info_mat(info)
                    noise_model = gtsam.noiseModel.Gaussian.Information(info_m)
                    graph.add(gtsam.BetweenFactorPose3(id_e1, id_e2 , gtsam.Pose3(rot_mat, t_vec), noise_model))
        
        # Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()

    resultPoses = gtsam.utilities.extractPose3(result)
    initialPoses = data["poses"]

    plot_traj_3d(initialPoses[:, 1:4], resultPoses[:, -3:], "solve_pose_slam_3d_incremental.png")
    plot_traj_2d(initialPoses[:, 1:3], resultPoses[:, -3:-1], "solve_pose_slam_3d_incremental_x_y.png")
    plot_traj_2d(initialPoses[:, 2:4], resultPoses[:, -2:], "solve_pose_slam_3d_incremental_y_z.png", [1, 2])

if __name__ == "__main__":
    # Problem 1
    data2d = load_g2o(INTEL_2D)
    solve_pose_slam_2d_batch(data2d, INTEL_2D)
    solve_pose_slam_2d_incremental(data2d)
    
    # Problem 2
    data3d = load_g2o(GARAGE_3D)
    solve_pose_slam_3d_batch(data3d, GARAGE_3D)
    solve_pose_slam_3d_incremental(data3d)

    # All optimized trajectory figures are saved under FIGURE_PATH
