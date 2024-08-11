import glob
import open3d as o3d

DATASETS = [
    "fractal20220817_data",
    "kuka",
    "bridge",
    "taco_play",
    "jaco_play",
    "berkeley_cable_routing",
    "roboturk",
    "nyu_door_opening_surprising_effectiveness",
    "viola",
    "berkeley_autolab_ur5",
    "toto",
    "language_table",
    "columbia_cairlab_pusht_real",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "nyu_rot_dataset_converted_externally_to_rlds",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "maniskill_dataset_converted_externally_to_rlds",
    "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "usc_cloth_sim_converted_externally_to_rlds",
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_saytap_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "robo_net",
    "berkeley_mvp_converted_externally_to_rlds",
    "berkeley_rpt_converted_externally_to_rlds",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "stanford_mask_vit_converted_externally_to_rlds",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "dlr_sara_pour_converted_externally_to_rlds",
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "asu_table_top_converted_externally_to_rlds",
    "stanford_robocook_converted_externally_to_rlds",
    "eth_agent_affordances",
    "imperialcollege_sawyer_wrist_cam",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "uiuc_d3field",
    "utaustin_mutex",
    "berkeley_fanuc_manipulation",
    "cmu_play_fusion",
    "cmu_stretch",
    "berkeley_gnm_recon",
    "berkeley_gnm_cory_hall",
    "berkeley_gnm_sac_son",
]

# The sorted datasets according to their size, ascending order.
DATASETS_SORTED = [
    "nyu_rot_dataset_converted_externally_to_rlds",
    "utokyo_saytap_converted_externally_to_rlds",
    "imperialcollege_sawyer_wrist_cam",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "usc_cloth_sim_converted_externally_to_rlds",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "cmu_stretch",
    "asu_table_top_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "berkeley_gnm_cory_hall",
    "austin_buds_dataset_converted_externally_to_rlds",
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
    "columbia_cairlab_pusht_real",
    "dlr_sara_pour_converted_externally_to_rlds",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "berkeley_cable_routing",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "cmu_play_fusion",
    "berkeley_gnm_sac_son" "nyu_door_opening_surprising_effectiveness",
    "berkeley_fanuc_manipulation",
    "jaco_play",
    "viola",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "berkeley_mvp_converted_externally_to_rlds",
    "uiuc_d3field",
    "eth_agent_affordances",
    "berkeley_gnm_recon",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "utaustin_mutex",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
]


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return [
        f"gs://gresearch/robotics/{dataset_name}/{version}",
        f"{dataset_name}",
        f"{version}",
    ]


def save_as_gif(images, path="temp.gif"):
    # Render the images as the gif:
    images[0].save(path, save_all=True, append_images=images[1:], duration=500, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes


def get_dataset_folder(dataset_name, dataset_dir):
    _, _, version = dataset2path(dataset_name)

    folder_candidates = glob.glob(dataset_dir + "*" + dataset_name)
    if not folder_candidates:
        return ""

    match_folder = folder_candidates[0] + f"\\{version}\\"
    return match_folder


def numpyarray_to_open3d(numpy_array):
    pcds = []
    id = 0
    for arr in numpy_array:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr)
        pcds.append(pcd)
        name = str(id).zfill(4) + "_cloud.pcd"
        id += 1
        o3d.io.write_point_cloud(name, pcd, print_progress=True)

    o3d.visualization.draw_geometries(pcds)
    return pcd


def get_feature_key(initial_key, features):
    if initial_key not in features:
        raise ValueError(
            f"The key {initial_key} was not found.\n"
            + "Please choose a different image key to display for this dataset.\n"
            + "Here is the observation spec:\n"
            + str(features)
        )

    if initial_key == "image" and "highres_image" in features:
        initial_key = "highres_image"

    return initial_key
