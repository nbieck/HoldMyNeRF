import copy
import random
import argparse
import os

import open3d as o3d
import numpy as np

def draw_results(m1, m2, t):
    m1t = copy.deepcopy(m1)
    m2t = copy.deepcopy(m2)

    m1t.paint_uniform_color([1, 0.7, 0])
    m2t.paint_uniform_color([0, 0.65, 0.93])
    m2t.transform(t)

    o3d.visualization.draw_geometries([m1t, m2t], width=1280, height=720)

# https://stackoverflow.com/questions/70160183/how-can-i-align-register-two-meshes-in-open3d-python
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def main():
    random.seed(0)

    voxel_size = 0.01

    parser = argparse.ArgumentParser()
    parser.add_argument("groundTruth")
    parser.add_argument("comparison")
    parser.add_argument("output")
    args = parser.parse_args()
    cwd = os.getcwd()

    groundTruth = os.path.join(cwd, args.groundTruth)
    resultMesh = os.path.join(cwd, args.comparison)

    truth_mesh = o3d.io.read_triangle_mesh(groundTruth)
    truth_mesh.compute_vertex_normals()
    res_mesh = o3d.io.read_triangle_mesh(resultMesh)
    res_mesh.compute_vertex_normals()

    #get rid of artefacts in pipeline output
    if True:
        clusters, n_tris, _ = (res_mesh.cluster_connected_triangles())
        clusters = np.asarray(clusters)
        n_tris = np.asarray(n_tris)

        largest_cluster = n_tris.argmax()
        to_remove = clusters != largest_cluster
        res_mesh.remove_triangles_by_mask(to_remove)
    draw_results(truth_mesh, res_mesh, np.identity(4))

    truth_cloud = truth_mesh.sample_points_uniformly(1000)
    res_cloud = res_mesh.sample_points_uniformly(1000)
    truth_box = truth_cloud.get_axis_aligned_bounding_box()
    res_box = res_cloud.get_axis_aligned_bounding_box()
    scale1 = truth_box.get_max_extent() / res_box.get_max_extent()
    res_mesh.scale(scale1, center=res_cloud.get_center())
    res_cloud.scale(scale1, center=res_cloud.get_center())

    draw_results(truth_cloud, res_cloud, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(res_cloud, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(truth_cloud, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_results(truth_cloud, res_cloud, result_ransac.transformation)

    threshold = voxel_size * 0.4
    reg_p2p = o3d.pipelines.registration.registration_icp(
        res_cloud, truth_cloud, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    print(reg_p2p)
    draw_results(truth_cloud, res_cloud, reg_p2p.transformation)

    res_mesh.transform(reg_p2p.transformation)
    res_cloud.transform(reg_p2p.transformation)
    truth_box = truth_cloud.get_axis_aligned_bounding_box()
    res_box = res_cloud.get_axis_aligned_bounding_box()
    scale2 = truth_box.get_max_extent() / res_box.get_max_extent()
    res_mesh.scale(scale2, center=res_cloud.get_center())
    res_cloud.scale(scale2, center=res_cloud.get_center())
    draw_results(truth_cloud, res_cloud, np.identity(4))

    draw_results(truth_mesh, res_mesh, np.identity(4))

    o3d.io.write_triangle_mesh(os.path.join(cwd, args.output), res_mesh)

if __name__ == "__main__":
    main()