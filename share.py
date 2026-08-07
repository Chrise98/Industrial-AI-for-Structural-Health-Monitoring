'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.
'''
#%%
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

project_folder = Path(__file__).resolve().parent

image_folder = project_folder / "images"
image_folder.mkdir(exist_ok=True)

ground_bin_width = 0.05
min_samples = 5


#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()

def get_ground_level(pcd, dataset_name):
    z_values = pcd[:, 2]

    minimum_z = np.min(z_values)
    maximum_z = np.max(z_values)

    bin_edges = np.arange(minimum_z,maximum_z + ground_bin_width,ground_bin_width)

    histogram_values, histogram_edges = np.histogram(z_values,bins=bin_edges)

    largest_bin_index = np.argmax(histogram_values)

    ground_level = (histogram_edges[largest_bin_index]+ histogram_edges[largest_bin_index + 1]) / 2

    plt.figure(figsize=(10, 6))
    plt.hist(z_values, bins=bin_edges, edgecolor="black", linewidth=0.2)
    plt.axvline(ground_level, linestyle="--", linewidth=2, label=f"Ground level = {ground_level:.2f} m")
    plt.title(f"{dataset_name}: histogram of z-values")
    plt.xlabel("z value (m)")
    plt.ylabel("Number of points")
    plt.legend()
    plt.savefig(image_folder / f"{dataset_name}_ground_histogram.png",dpi=300,bbox_inches="tight")
    plt.show()

    return float(ground_level)

def get_optimal_epsilon(points, dataset_name):
    tree = KDTree(points)

    distances, _ = tree.query(
        points,
        k=min_samples
    )

    fifth_neighbour_distances = np.sort(
        distances[:, -1]
    )

    # Remove only the most extreme final 0.5 percent
    # while locating the elbow.
    trim_count = int(
        len(fifth_neighbour_distances) * 0.995
    )

    trimmed_distances = fifth_neighbour_distances[
        :trim_count
    ]

    x_normalized = np.linspace(
        0,
        1,
        len(trimmed_distances)
    )

    y_normalized = (
        trimmed_distances - trimmed_distances.min()
    ) / (
        trimmed_distances.max()
        - trimmed_distances.min()
    )

    difference = x_normalized - y_normalized

    elbow_index = np.argmax(difference)

    optimal_epsilon = float(
        trimmed_distances[elbow_index]
    )

    optimal_epsilon = round(
        optimal_epsilon,
        2
    )

    plt.figure(figsize=(10, 6))
    plt.plot(trimmed_distances)
    plt.axvline(elbow_index,linestyle="--",label="Detected elbow")
    plt.axhline(optimal_epsilon, linestyle="--",label=f"eps = {optimal_epsilon:.2f}")
    plt.scatter(elbow_index,optimal_epsilon,s=60)
    plt.title(f"{dataset_name}: fifth-neighbour elbow plot")

    plt.xlabel("Points sorted by distance")
    plt.ylabel("Fifth-neighbour distance")
    plt.legend()

    plt.savefig(image_folder / f"{dataset_name}_elbow.png", dpi=300, bbox_inches="tight")
    plt.show()

    return optimal_epsilon

def find_largest_cluster(
    pcd_above_ground,
    cluster_labels
):
    largest_cluster_information = None

    for cluster_number in np.unique(
        cluster_labels
    ):

        # DBSCAN assigns -1 to noise.
        if cluster_number == -1:
            continue

        cluster_points = pcd_above_ground[
            cluster_labels == cluster_number
        ]

        minimum_x = np.min(
            cluster_points[:, 0]
        )

        maximum_x = np.max(
            cluster_points[:, 0]
        )

        minimum_y = np.min(
            cluster_points[:, 1]
        )

        maximum_y = np.max(
            cluster_points[:, 1]
        )

        x_span = maximum_x - minimum_x
        y_span = maximum_y - minimum_y

        cluster_area = x_span * y_span

        current_information = {
            "cluster_number": int(cluster_number),
            "points": cluster_points,
            "minimum_x": float(minimum_x),
            "maximum_x": float(maximum_x),
            "minimum_y": float(minimum_y),
            "maximum_y": float(maximum_y),
            "x_span": float(x_span),
            "y_span": float(y_span),
            "area": float(cluster_area)
        }

        if (
            largest_cluster_information is None
            or current_information["area"]
            > largest_cluster_information["area"]
        ):
            largest_cluster_information = (
                current_information
            )

    return largest_cluster_information

#%% read file containing point cloud data
datasets = [("dataset1", "dataset1.npy"),("dataset2", "dataset2.npy")]

for dataset_name, dataset_file in datasets:
    print("\n" + "=" * 60)
    print("Processing:", dataset_name)
    print("=" * 60)

    # Load one dataset
    pcd = np.load(dataset_file)

    print("Point-cloud shape:", pcd.shape)

    # Show the original point cloud
    show_cloud(pcd[::10])


#%% show downsampled data in external window
#%matplotlib qt
#show_cloud(pcd[::10]) # keep every 10th point
#%% remove ground plane

# Task 1 (3)
#vfind the best value for the ground level
#VOne way to do it is useing a histogram 
#vnp.histogram
#update the function get_ground_level() with your changes
#For both the datasets
#Report the ground level in the readme file in your github project
#vAdd the histogram plots to your project readme


    est_ground_level = get_ground_level(pcd,dataset_name)

    print(f"Ground level: {est_ground_level:.2f} m")

    pcd_above_ground = pcd[pcd[:, 2] > est_ground_level]

    print("Points above ground:",pcd_above_ground.shape)

    # Show the cloud after removing the ground
    show_cloud(pcd_above_ground[::5])


#%%

#Task 2 (+1)

#Find an optimized value for eps.
#Plot the elbow and extract the optimal value from the plot
#Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

#https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
#https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

#For both the datasets
#Report the optimal value of eps in the Readme to your github project
#Add the elbow plots to your github project Readme
#Add the cluster plots to your github project Readme
#'''

    optimal_eps = get_optimal_epsilon(pcd_above_ground,dataset_name)

    print(f"Optimal epsilon: {optimal_eps:.2f}")

    clustering = DBSCAN(eps=optimal_eps,min_samples=min_samples).fit(pcd_above_ground)

    clusters = (len(set(clustering.labels_))- (1 if -1 in clustering.labels_ else 0))

    print("Number of clusters:", clusters)
    print("Number of noise points:",np.sum(clustering.labels_ == -1))

    colors = [plt.cm.Spectral(each)for each in np.linspace(0,1,max(clusters, 1))]

    # Plot resulting clusters
    plt.figure(figsize=(10, 10))

    plt.scatter(
        pcd_above_ground[:, 0],
        pcd_above_ground[:, 1],
        c=clustering.labels_,
        cmap=matplotlib.colors.ListedColormap(colors),
        s=2
    )

    plt.title(f"{dataset_name}: DBSCAN - {clusters} clusters",fontsize=20)
    plt.xlabel("x axis", fontsize=14)
    plt.ylabel("y axis", fontsize=14)
    plt.savefig(image_folder / f"{dataset_name}_clusters.png",dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()

#%%
#'''
#Task 3 (+1)
#Find the largest cluster, since that should be the catenary, 
#beware of the noise cluster.
#Use the x,y span for the clusters to find the largest cluster
#For both the datasets
#Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
#Add the plot of the catenary cluster to the readme
#'''

    catenary = find_largest_cluster(
        pcd_above_ground,
        clustering.labels_
    )

    print("Catenary cluster:",catenary["cluster_number"])

    print(f"min(x) = {catenary['minimum_x']:.2f} m")
    print(f"min(y) = {catenary['minimum_y']:.2f} m")
    print(f"max(x) = {catenary['maximum_x']:.2f} m")
    print(f"max(y) = {catenary['maximum_y']:.2f} m")

    print(f"x span = {catenary['x_span']:.2f} m")
    print(f"y span = {catenary['y_span']:.2f} m")
    print(f"Catenary area = {catenary['area']:.2f} m²")

    plt.figure(figsize=(10, 8))
    plt.scatter(catenary["points"][:, 0],catenary["points"][:, 1],s=2)
    plt.title(f"{dataset_name}: selected catenary cluster\n" f"Area = {catenary['area']:.2f} m²")
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.savefig(image_folder/ f"{dataset_name}_catenary_cluster.png",dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()