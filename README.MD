# Analyze RTX dataset
  
## Pre-requisitions

### Setup with conda ( :+1: **Recommanded**  )

`$ conda env create -f environment.yml -n myenv`


### Manually Setup ( :-1: *Not Recommanded*, skip if use conda)
Install Tensorflow:

 `$ pip install --force-reinstall -v "tensorflow==2.17.0"`

Install open3d for pointcloud processing:

 `$ pip install --force-reinstall -v "open3d==0.18.0"`

Install tensorflow-datasets to access local or online datasets:

`$ pip install --force-reinstall -v "tensorflow-datasets==4.9.6"`

## Investigate a certain dataset
`$ python feature_fetcher.py`

## Datasets in details
|dataset name |feature | size | in use | image |
|-|-|-|:-:|-|
| nyu_rot_dataset                 | image | 84*84   |       | ![image](./example/0_nyu_rot_dataset_converted_externally_to_rlds.gif)|
| utokyo_saytap                   | image |         |       | ![image](./example/1_utokyo_saytap_converted_externally_to_rlds.gif)|
| imperialcollege_sawyer_wrist_cam| image | 64*64   |       | ![image](./example/2_imperialcollege_sawyer_wrist_cam.gif)|
| utokyo_xarm_bimanual            | image | 256*256 |       | ![image](./example/3_utokyo_xarm_bimanual_converted_externally_to_rlds.gif)|
| usc_cloth_sim                   | image | 32*32   |       | ![image](./example/4_usc_cloth_sim_converted_externally_to_rlds.gif)|
| tokyo_u_lsmo                    | image | 120*120 |       | ![image](./example/5_tokyo_u_lsmo_converted_externally_to_rlds.gif)|
| utokyo_pr2_opening_fridge       | image | 128*128 |       | ![image](./example/6_utokyo_pr2_opening_fridge_converted_externally_to_rlds.gif)|
| cmu_franka_exploration_dataset  | image | 480*480 | :+1:  | ![image](./example/7_cmu_franka_exploration_dataset_converted_externally_to_rlds.gif)|
| cmu_stretch                     | image | 128*128 |       | ![image](./example/8_cmu_stretch.gif)|
| asu_table_top                   | image | 224*224 |       | ![image](./example/9_asu_table_top_converted_externally_to_rlds.gif)|
| utokyo_pr2_tabletop_manipulation| image | 128*128 |       | ![image](./example/10_utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds.gif)|
| utokyo_xarm_pick_and_place      | image | 224*224 |       | ![image](./example/11_utokyo_xarm_pick_and_place_converted_externally_to_rlds.gif)|
| ucsd_kitchen_dataset            | image | 640*480 | :+1:  | ![image](./example/12_ucsd_kitchen_dataset_converted_externally_to_rlds.gif)|
| berkeley_gnm_cory_hall          | image | 85*64   |       | ![image](./example/13_berkeley_gnm_cory_hall.gif)|
| austin_buds_dataset             | image | 128*128 |       | ![image](./example/14_austin_buds_dataset_converted_externally_to_rlds.gif)|
| dlr_sara_grid_clamp             | image | 640*480 | :+1:  | ![image](./example/15_dlr_sara_grid_clamp_converted_externally_to_rlds.gif)|
| columbia_cairlab_pusht_real     | image | 320*240 |       | ![image](./example/16_columbia_cairlab_pusht_real.gif)|
| dlr_sara_pour                   | image | 640*480 | :+1:  | ![image](./example/17_dlr_sara_pour_converted_externally_to_rlds.gif)|
| dlr_edan_shared_control         | image | 640*360 | :+1:  | ![image](./example/18_dlr_edan_shared_control_converted_externally_to_rlds.gif)|
| ucsd_pick_and_place_dataset     | image | 224*224 |       | ![image](./example/19_ucsd_pick_and_place_dataset_converted_externally_to_rlds.gif)|
| berkeley_cable_routing          | image | 128*128 |       | ![image](./example/20_berkeley_cable_routing.gif)|
| nyu_franka_play_dataset         | image | 128*128 |       | ![image](./example/21_nyu_franka_play_dataset_converted_externally_to_rlds.gif)|
| austin_sirius_dataset           | image | 84*84   |       | ![image](./example/22_austin_sirius_dataset_converted_externally_to_rlds.gif)|
| cmu_play_fusion                 | image | 128*128 |       | ![image](./example/23_cmu_play_fusion.gif)|
| berkeley_fanuc_manipulation     | image | 224*224 |       | ![image](./example/25_berkeley_fanuc_manipulation.gif)|
| jaco_play                       | image | 224*224 |       | ![image](./example/26_jaco_play.gif)|
| kaist_nonprehensile             | image | 640*480 | :+1:  | ![image](./example/28_kaist_nonprehensile_converted_externally_to_rlds.gif)|
| eth_agent_affordances           | LIDAR | NA      |       | ![image](./example/31_eth_agent_affordances.gif)|
| berkeley_gnm_recon              | image | 160*120 |       | ![image](./example/32_berkeley_gnm_recon.gif)|
| austin_sailor_dataset           | image | 128*128 |       | ![image](./example/33_austin_sailor_dataset_converted_externally_to_rlds.gif)|
| utaustin_mutex                  | image | 128*128 |       | ![image](./example/34_utaustin_mutex.gif)|
| stanford_kuka_multimodal        | image | 128*128 |       | ![image](./example/35_stanford_kuka_multimodal_dataset_converted_externally_to_rlds.gif)|
