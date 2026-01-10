import sys
print(sys.executable)


import supervision as sv

import sys
import numpy as np
import glob
import os
import bz2
import pandas as pd
import geopandas as gpd
import pickle
from PIL import Image
import street_triangulation as tri
import supervision as sv
import cv2
import math
import pdb
import contextily as ctx
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None)

# sys.path.append(r'E:\Research\StreetView\gsv_pano')
sys.path.append(r'D:\Research\StreetView\gsv_pano')

from pano import GSV_pano
import utils

detect_dir = r'D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\street_image_mapping\Heyward_st_pano_thumnail_segmented'
depthmap_dir = r'D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\street_image_mapping\Heyward_st_pano_thumnail_depth'
image_dir = r'D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\street_image_mapping\Heyward_st_pano_thumnail'
pano_meta_fname = r"D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\street_image_mapping\Heyward_st_pano_thumnail.csv"
annotated_dir = r'D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\street_image_mapping\Heyward_st_pano_thumnail_segmented'
local_crs = 6569 # EPSG:6569, NAD83(2011) / South Carolina
save_dir = r'D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\street_image_mapping\Heyward_st_trunks'

img_height = 768
img_width = 1024

camera_dis_thres = 20
trunk_near_thres = 3


os.makedirs(save_dir, exist_ok=True)


CLASSES = ['tree_trunk']   # , 'tree root collar'
trunk_index = 0

pano_meta_df = pd.read_csv(pano_meta_fname).set_index('panoId')

detect_files = glob.glob(os.path.join(detect_dir, "*.bz2"))
detect_files = sorted(detect_files)
print("Found file count:", len(detect_files))
# ['D:\\OneDrive_PSU\\OneDrive - The Pennsylvania State University\\Research_doc\\street_image_mapping\\Heyward_st_pano_thumnail_segmented\\piIyY_Bgq2dmllHQlN7FRw_0.0_0.0_0_226.73_R135.pkl.bz2']
detect_files = ['D:\\OneDrive_PSU\\OneDrive - The Pennsylvania State University\\Research_doc\\street_image_mapping\\Heyward_st_pano_thumnail_segmented\\piIyY_Bgq2dmllHQlN7FRw_0.0_0.0_0_226.73_R135.pkl.bz2', 'D:\\OneDrive_PSU\\OneDrive - The Pennsylvania State University\\Research_doc\\street_image_mapping\\Heyward_st_pano_thumnail_segmented\\XShHFvEc3Mi2fhPmjY7OGw_0.0_0.0_0_180.97_R90.pkl.bz2'] + detect_files  # add a file on the top for testing
# detect_files
print("Found file count:", len(detect_files))


pair_trunk_df_list = []

for idx, detect_file in enumerate(detect_files[:]): # 458:460
    try:
        # print("detect_file: ", detect_file)
        # detect_file = os.path.join(detect_dir, r'l-PhEd7ACdyR12n08p1Q0Q_0.0_0.0_0_33.59_R90.pkl.bz2')  # [68:69]
        # detect_file = os.path.join(detect_dir, r'XciMXMjYPg0jhKh8cKG1eA_0.0_0.0_0_78.39_R135.pkl.bz2')
        gdf_list = []

        basename = os.path.basename(detect_file).replace('.pkl.bz2', '.jpg')
        direction_str = basename.replace(".pkl.bz2", "").split("_")[-1]

        image_fname = os.path.join(image_dir, basename)

        if direction_str in ['L135', 'R135']:  #  I make a mistake when naming the thumbnails: the "L135" and "L45" should be switched.
            print("skip: ", direction_str, basename)

        # pdb.set_trace()

        print(f"Processing: {idx + 1} / {len(detect_files)}: {basename}" )


        # find image pairs
        panoID = basename[:22]
        pano_meta = pano_meta_df.loc[panoID]
        # print("pano_meta:", pano_meta)
        image_pairs = tri.form_image_pairs(detect_file, pano_meta, detect_dir=None)
        # print("idx:", idx)


        if len(image_pairs) == 0:
            print("No image_pairs, skip. \n")
            continue

        else:
            for img1, img2 in image_pairs:
                print(f"    image pairs: {os.path.basename(img1), os.path.basename(img2)}")

        print()
        # print("idx:", idx)
        trunk_df = tri.get_trunks_in_an_image(detect_file, trunk_index, image_dir, pano_meta_df, depthmap_dir, local_crs)
        # print("idx:", idx, trunk_df)  # passed

        for pair in image_pairs:
            print("pair[1]:", pair[1])
            paired_trunk_df = tri.get_trunks_in_an_image(pair[1], trunk_index, image_dir, pano_meta_df, depthmap_dir, local_crs)
            print("paired_trunk_df:", paired_trunk_df)
        all_gdf = tri.merge_trunk_df_pair(trunk_df, paired_trunk_df, local_crs)
        print("all_gdf:", all_gdf)

        all_trunk_gdf = all_gdf.query("Type == 'trunk' ")

        all_camera_gdf = all_gdf.query("Type == 'camera' ")

        trunk_pairs_df = tri.form_trunk_pairs(all_trunk_gdf, near_thres=trunk_near_thres, camera_dis_thres=camera_dis_thres)

        print("trunk_pairs_df:", trunk_pairs_df)

        pair_cnt = int(len(trunk_pairs_df) / 2)

        # print("all_trunk_gdf:", all_trunk_gdf)

        for p in range(pair_cnt):
            a_pair_trunk_df = trunk_pairs_df.iloc[2 * p:2 * p + 2].copy()
            a_pair_trunk_df = tri.triangulate_trunk_location(a_pair_trunk_df)
            a_pair_trunk_df = tri.compute_diameter(a_pair_trunk_df)

            pair_trunk_df_list.append(a_pair_trunk_df)

            tri.show_detailed_images(a_pair_trunk_df, local_crs, image_dir, annotated_dir, save_dir)

        print()

    except Exception as e:
        print()
        print("Error:", e, idx, basename)
        print()
        break

all_pair_trunk_df = pd.concat(pair_trunk_df_list)
csv_fname = os.path.join(save_dir, "detected_trunks.csv")
all_pair_trunk_df.to_csv(csv_fname, index=False)

geometry = gpd.points_from_xy(all_pair_trunk_df['tri_x'], all_pair_trunk_df['tri_y'])
localized_trunk_gdf = gpd.GeoDataFrame(data=all_pair_trunk_df, geometry=geometry).set_crs(local_crs)
localized_trunk_fname = os.path.join(save_dir, "localized_trunks.shp")
localized_trunk_gdf.to_file(localized_trunk_fname)




