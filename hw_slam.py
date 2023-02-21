import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gtsam

# Path to data folder
DATA_PATH = "data/"
FIGURE_PATH = "figures/"
DTYPE = np.float64


def load_g2o(filename:str)->dict:
    """
    Load a G2O file. 
    Each VERTEX will be stored in "poses" ndarray and each EDGE will be stored in "edges" ndarray.

    Return: a data dict contain poses and edges.
    """
    poses = []
    edges = []
    path_to_file = os.path.join(DATA_PATH, filename)
    with open(path_to_file, 'r') as f:
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

def construct_info_mat(info_v:list)->np.ndarray:
    """
    Construct a information matrix from a list of its upper triangular entries.
    Only 2D and 3D spaces are supported.
    """
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

def plot_traj_2d(before:np.ndarray, after:np.ndarray, filename="test2d.png", axis_idx=[0, 1], show=False):
    """
    Plot a 2D figure of trajectories before and after optimization.
    """
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
    # plt.title(filename[:-4].replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, filename))
    if show:
        plt.show()

def plot_traj_3d(before:np.ndarray, after:np.ndarray, filename="test3d.png", show=False):
    """
    Plot a 3D figure of trajectories before and after optimization.
    """
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
    # plt.title(filename[:-4].replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, filename))
    if show:
        plt.show()

def solve_pose_slam_2d_batch(data_name:str, figure_name:str):
    """
    Use the Gauss-Newton solver to solve a 2D pose graph SLAM in a batch.
    """
    data = load_g2o(data_name)

    is3D = False
    graph, initial = gtsam.readG2o(os.path.join(DATA_PATH, data_name), is3D)

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
    
    plot_traj_2d(initialPoses[:, 1:3], resultPoses[:, :2], figure_name)

def solve_pose_slam_2d_incremental(data_name:str, figure_name:str):
    """
    Use the incremental Smoothing and Mapping method to solve a 2D pose graph SLAM.
    """
    data = load_g2o(data_name)

    # Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
    # update calls are required to perform the relinearization.
    params = gtsam.ISAM2Params()
    # Only relinearize variables whose linear delta magnitude is greater than this threshold
    params.setRelinearizeThreshold(0.1)
    # Only relinearize any variables every relinearizeSkip calls to ISAM2::update
    params.setRelinearizeSkip(10)
    isam = gtsam.ISAM2(params)

    # Define the prior factor to the factor graph
    priorModel = gtsam.noiseModel.Diagonal.Variances(gtsam.Point3(1e-6, 1e-6, 1e-8))

    # Data initialize
    poses = data["poses"]
    odometry_measurements = data["edges"]

    # Initialize the current estimate which is used during the incremental inference loop.
    result = None
    # Create a Nonlinear factor graph as well as the data structure to hold state estimates.
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    for pose in tqdm(poses, desc="Processing"):
        
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
        graph.resize(0)
        initial_estimate.clear()

    is3D = False
    graph_batch, initial = gtsam.readG2o(os.path.join(DATA_PATH, data_name), is3D)
    print("Optimization complete")
    print("initial error = ", graph_batch.error(initial))
    print("final error = ", graph_batch.error(result))

    resultPoses = gtsam.utilities.extractPose2(result)
    initialPoses = data["poses"]

    plot_traj_2d(initialPoses[:, 1:3], resultPoses[:, :2], figure_name)

def solve_pose_slam_3d_batch(data_name:str, figure_name:str):
    """
    Use the Gauss-Newton solver to solve a 3D pose graph SLAM in a batch.
    """
    data = load_g2o(data_name)

    is3D = True
    graph, initial = gtsam.readG2o(os.path.join(DATA_PATH, data_name), is3D)

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
    
    plot_traj_3d(initialPoses[:, 1:4], resultPoses[:, -3:], figure_name)
    plot_traj_2d(initialPoses[:, 1:3], resultPoses[:, -3:-1], figure_name[:-4]+"_x_y.png")
    plot_traj_2d(initialPoses[:, 2:4], resultPoses[:, -2:], figure_name[:-4]+"_y_z.png", [1, 2])

def solve_pose_slam_3d_incremental(data_name:str, figure_name:str):
    """
    Use the incremental Smoothing and Mapping method to solve a 3D pose graph SLAM.
    """
    data = load_g2o(data_name)

    # Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
    # update calls are required to perform the relinearization.
    params = gtsam.ISAM2Params()
    # Only relinearize variables whose linear delta magnitude is greater than this threshold
    params.setRelinearizeThreshold(0.1)
    # Only relinearize any variables every relinearizeSkip calls to ISAM2::update
    params.setRelinearizeSkip(10)
    isam = gtsam.ISAM2(params)

    # Define the prior factor to the factor graph
    priorModel = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4], dtype=DTYPE))
    # Data initialize
    poses = data["poses"]
    odometry_measurements = data["edges"]

    # Initialize the current estimate which is used during the incremental inference loop.
    result = None
    # Create a Nonlinear factor graph as well as the data structure to hold state estimates.
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    for pose in tqdm(poses, desc="Processing"):
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
        graph.resize(0)
        initial_estimate.clear()
    
    is3D = True
    graph_batch, initial = gtsam.readG2o(os.path.join(DATA_PATH, data_name), is3D)
    print("Optimization complete")
    print("initial error = ", graph_batch.error(initial))
    print("final error = ", graph_batch.error(result))

    resultPoses = gtsam.utilities.extractPose3(result)
    initialPoses = data["poses"]

    plot_traj_3d(initialPoses[:, 1:4], resultPoses[:, -3:], figure_name)
    plot_traj_2d(initialPoses[:, 1:3], resultPoses[:, -3:-1], figure_name[:-4]+"_x_y.png")
    plot_traj_2d(initialPoses[:, 2:4], resultPoses[:, -2:], figure_name[:-4]+"_y_z.png", [1, 2])

if __name__ == "__main__":
    # Dataset path
    intel_2d = "input_INTEL_g2o.g2o"
    garage_3d = "parking-garage.g2o"

    # Problem 1
    solve_pose_slam_2d_batch(intel_2d, "solve_pose_slam_2d_batch.png")
    solve_pose_slam_2d_incremental(intel_2d, "solve_pose_slam_2d_incremental.png")
    
    # # Problem 2
    solve_pose_slam_3d_batch(garage_3d, "solve_pose_slam_3d_batch.png")
    solve_pose_slam_3d_incremental(garage_3d, "solve_pose_slam_3d_incremental.png")

    # All optimized trajectory figures are saved under FIGURE_PATH