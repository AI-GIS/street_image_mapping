import supervision as sv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np
import glob
import os
import bz2
import pandas as pd
import geopandas as gpd
import pickle
from PIL import Image
import supervision as sv
import cv2
import math
from collections import Counter
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import matplotlib.lines as mlines
sys.path.append(r'E:\Research\StreetView\gsv_pano')
sys.path.append(r'D:\Research\StreetView\gsv_pano')

from pano import GSV_pano
import utils
import contextily as ctx


CLASSES = ['tree_trunk', 'vehicle', 'person', 'sidewalk', 'road', 'building', 'grass']   # , 'tree root collar'


def get_mask_from_detection(image, detections):
    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )


def show_detection_segment(image, detections):
# annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    # print(detections)
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # %matplotlib inline
    sv.plot_image(annotated_image, (16, 16))

    return annotated_image

def clean_none(detection):   # not necessary, need to change the processing code (after detecting boxes)
    # detections = detections[detections.class_id != None]
    class_ids= detection.class_id
    # print(class_ids)
    detection.xyxy = detection.xyxy[class_ids != None]
    detection.mask = detection.mask[class_ids != None]    
    detection.confidence = detection.confidence[class_ids != None]
    detection.class_id = detection.class_id[class_ids != None]
    return detection

def get_root_distance_from_depthmap(depthmap, detection, trunk_id, root_ground_height=10):
    '''
    root_ground_height: extract the pixels in these rows from depthmap, 
    and take their average as the distance.
    '''
    trunk_box = detection[1].xyxy[trunk_id]
    trunk_mask = detection[1].mask[trunk_id]
    idx_row, idx_col = np.where(trunk_mask)
    row_col_idx = np.stack([idx_row, idx_col], axis=1)
    # root_ground_height = 10 # unit: row

    bottom_row = idx_row.max()
    top_row = max(idx_row.max() - root_ground_height, 0)
    root_row_idx = np.where(np.logical_and(top_row < idx_row, idx_row < bottom_row))

    # the distance of from camera to trunk bottom.
    root_idx = row_col_idx[root_row_idx]
    distance = depthmap[root_idx[:, 0], root_idx[:, 1]].mean()
    return distance


def get_trunk_property(depthmap, detection, trunk_id, 
                        horizontal_fov_rad=math.radians(90), 
                        root_ground_height=10):   
    trunk = {}

    trunk_box = detection[1].xyxy[trunk_id]
    trunk_mask = detection[1].mask[trunk_id]
    idx_row, idx_col = np.where(trunk_mask)
    row_col_idx = np.stack([idx_row, idx_col], axis=1)
    # root_ground_height = 10 # unit: row

    bottom_row = idx_row.max()
    top_row = max(idx_row.max() - root_ground_height, 0)
    root_row_idx = np.where(np.logical_and(top_row < idx_row, idx_row < bottom_row))

    # the distance of from camera to trunk bottom.
    root_idx = row_col_idx[root_row_idx]
    distance = depthmap[root_idx[:, 0], root_idx[:, 1]].mean()
    trunk['distance'] = distance

    # get the center column
    image_mid_row = depthmap.shape[0] / 2
    image_mid_row = int(image_mid_row)
    bottom_part_row_idx = np.where(idx_row > image_mid_row)
    mask_bottom_part_idx = row_col_idx[bottom_part_row_idx]  # idx in raw image row/col

    center_col = np.median(mask_bottom_part_idx[:, 1])
    trunk['root_center_col'] = int(center_col)

    center_col_row_idx = np.where(mask_bottom_part_idx[:, 1] == trunk['root_center_col'])
    center_col_rows = mask_bottom_part_idx[center_col_row_idx][:, 0]

    trunk['root_bottom_row'] = int(center_col_rows.max())

    azimuth_rad, altitude_rad = utils.row_col_to_angle(row=trunk['root_bottom_row'], 
                                                       col=trunk['root_center_col'], 
                                                        width=depthmap.shape[1], 
                                                        height=depthmap.shape[0], 
                                                        horizontal_fov_rad=horizontal_fov_rad)
    trunk['azimuth_rad'] = azimuth_rad
    trunk['altitude_rad'] = altitude_rad

    trunk['depth_h_distance'] = trunk['distance'] * math.cos(trunk['altitude_rad'])

    # get the breath height
    # find the trunk median
    # print("mask_bottom_part_idx.shape:", mask_bottom_part_idx.shape)
    # print("mask_bottom_part_idx:", mask_bottom_part_idx)
    # print("mask_bottom_part_idx[:, 0]:", mask_bottom_part_idx[:, 0])
    row_widths = Counter(mask_bottom_part_idx[:, 0])
    sorted_items = sorted(row_widths.items())
    n = len(sorted_items)
    median_key, median_value = sorted_items[n // 2]
    # print("row_widths:", row_widths)
    trunk['median_wid'] = np.median(median_value)
    trunk['median_wid_row'] = int(np.median(median_key))

    # find the start/end col for the median width row
    median_row_cols = mask_bottom_part_idx[mask_bottom_part_idx[:, 0] == trunk['median_wid_row']][:, 1]
    # print()
    trunk['median_row_start_col'] = median_row_cols.min()
    # trunk['median_row_end_col'] = trunk['median_row_start_col'] + trunk['median_wid']
    trunk['median_row_end_col'] = median_row_cols.max()


    return trunk

def form_image_pairs(detect_file, pano_meta, detect_dir=None):

    if not detect_dir:
        detect_dir = os.path.dirname(detect_file)

    basename = os.path.basename(detect_file)

    # only process the forward direction images
    direction_str = basename.replace(".pkl.bz2", "").split("_")[-1]
    # I make a mistake when naming the thumbnails: the "L135" and "L45" should be switched.
    # print(direction_str)
    if direction_str in ['L45', 'R135']:
        print("skip the backward thumbnail: ", direction_str, basename)
        # print()
        return []

    #pano_meta = pano_meta_df.loc[panoID]#.to_dict()   
    # find the forward panorama
    link_f_panoId = pano_meta['link_f_panoId']



    # form image pairs
    pairs = []
    direction = direction_str[0]
    direction_angle = int(direction_str[1:])
    # print("direction, direction_angle:", direction, direction_angle)

    if direction == "R":

        forward_R135_basename = link_f_panoId + "*R135.pkl.bz2"

        forward_R135_files = glob.glob(os.path.join(detect_dir, forward_R135_basename))
        if len(forward_R135_files) == 0:
            # return pairs
            pass
        else:
            forward_R135_file = forward_R135_files[0]

        if direction_angle == 90:
            # pair 1: R90 -- Forward_R135,
            if len(forward_R135_files) == 1:
                pairs.append([detect_file, forward_R135_file])

        if direction_angle == 45:
            # pair 1: R45 -- Forward_R135,
            if len(forward_R135_files) == 1:
                pairs.append([detect_file, forward_R135_file])
            # code line repeats, but can be undertood easier.

            # pair 2: R45 -- Forward_R90,     
            forward_R90_basename = link_f_panoId + "*R90.pkl.bz2"
            forward_R90_files = glob.glob(os.path.join(detect_dir, forward_R90_basename))
            # print("Forward R90 file:", forward_R90_file)
            if len(forward_R90_files) == 1:
                pairs.append([detect_file, forward_R90_files[0]])
 
    if direction == "L":
        forward_L135_basename = link_f_panoId + "*L45.pkl.bz2"       #!!!!!!!!! BUG !!!!!!!   should be 135
        # I make a mistake when naming the thumbnails: the "L135" and "L45" should be switched.

        forward_L135_files = glob.glob(os.path.join(detect_dir, forward_L135_basename))
        # print("Forward L135 file:", forward_L135_file)
        if len(forward_L135_files) == 0:
            # return pairs
            pass
        else:
            forward_L135_file = forward_L135_files[0]

        if direction_angle == 90:
            # pair 1: L90 -- Forward_L135
            if len(forward_L135_files) == 1:
                pairs.append([detect_file, forward_L135_file])

        # if direction_angle == 135:         #!!!!!!!!! BUG !!!!!!!   should be 45
        if direction_angle == 45:         #!!!!!!!!! BUG !!!!!!!   should be 45
            # I make a mistake when naming the thumbnails: the "L135" and "L45" should be switched.
            # pair 1: L45 -- Forward_L135
            if len(forward_L135_files) == 1:
                pairs.append([detect_file, forward_L135_file])
            # code line repeats, but can be undertood easier.

            # pair 2: L45 -- Forward_L90,     
            forward_L90_basename = link_f_panoId + "*L90.pkl.bz2"
            forward_L90_files = glob.glob(os.path.join(detect_dir, forward_L90_basename))
            # print("Forward L135 file:", forward_L135_file)
            if len(forward_L90_files) == 0:
                return pairs
            else:
                forward_L90_file = forward_L90_files[0]
            if len(forward_L90_files) == 1:
                # print("Forward L90 file:", forward_L90_file)
                pairs.append([detect_file, forward_L90_file])
    # print()
    return pairs

 
def remove_thumbnail_near_trunk(df, near_thres=10):
    '''
    near_thres: pixel unit
    '''
    df = df.sort_values('root_center_col')
    keep_idx = [True] * len(df)
    for i in range(len(df) - 1):
        diff = df.iloc[i]['root_center_col'] - df.iloc[i + 1]['root_center_col']
        if abs(diff) < near_thres:
            keep_idx[i + 1] = False
    return df[keep_idx]    


def get_trunks_in_an_image(detect_file, trunk_class_id, image_dir, pano_meta_df, depthmap_dir, local_crs):
    basename = os.path.basename(detect_file)
    heading_deg = basename.split("_")[-2]
    heading_deg = float(heading_deg)
    heading_rad = heading_deg / 180 * np.pi

    panoID = basename[:22]

    pano_obj = GSV_pano(panoId=panoID, crs_local=local_crs)

     
    direction_str = basename.replace(".pkl.bz2", "").split("_")[-1]



    with bz2.open(detect_file, 'rb') as f:
        detection = pickle.load(f)
        clean_none(detection[1])


    # Load trunks
    trunk_ids = np.where(detection[1].class_id == trunk_class_id)[0]
    # print(trunk_ids)

    basename = os.path.basename(detect_file).replace('.pkl.bz2', '_kitti_dpt_depthmap.png')
    depthmap_fname = os.path.join(depthmap_dir, basename)
    # print(depthmap_fname)
    depthmap = cv2.imread(depthmap_fname)[:, :, 0] / 5  # I saved the depthmap by multiply 5.
    # print(depthmap)
    trunk_list = []
    for trunk_id in trunk_ids:
        # print(trunk_id)

        trunk = get_trunk_property(depthmap, detection, trunk_id, root_ground_height=10)

        trunk['panoID'] = panoID

        pano_obj.calculate_xy()

        trunk['depth_x'] = pano_obj.x + trunk['depth_h_distance'] * math.cos(np.pi/2 - (heading_rad + trunk['azimuth_rad']))
        trunk['depth_y'] = pano_obj.y + trunk['depth_h_distance'] * math.sin(np.pi/2 - (heading_rad + trunk['azimuth_rad']))
        trunk['pano_x'] = pano_obj.x
        trunk['pano_y'] = pano_obj.y
        trunk['pano_heading_deg'] = pano_obj.jdata['Projection']['pano_yaw_deg']
        # print("pano_obj:", pano_obj.jdata['Projection']['pano_yaw_deg'])

        trunk['thumbnail_heading_deg'] = heading_deg
        trunk['trunk_heading_deg'] = np.degrees(trunk['azimuth_rad'] + heading_rad)
        if trunk['trunk_heading_deg'] > 360:
            trunk['trunk_heading_deg'] - trunk['trunk_heading_deg'] - 360
        trunk['image_file'] = basename.replace('_kitti_dpt_depthmap.png', '.jpg')

        trunk_list.append(trunk)

        # print("trunk:", trunk)
 
    # trunk_list = trunk_list[1:2] # sample a row   
    # trunk_list.append({"x":pano_obj.x, "y":pano_obj.y})
    trunk_df = pd.DataFrame(trunk_list).iloc[:]
    # trunk_df['type'] = 'trunk'
    trunk_df['azimuth_deg'] = np.degrees(trunk_df['azimuth_rad'])
    trunk_df['azimuth_deg'] = np.degrees(trunk_df['azimuth_rad'])
    # trunk_df.iloc[-1, trunk_df.columns.get_loc('type')] = 'camera'
 
    trunk_df = remove_thumbnail_near_trunk(trunk_df)

    return trunk_df


def draw_yaw_arrows(ax, paired_trunk_df, pano_len=3, trunk_len=20, width=0.3, extend=30):
    '''
    length:
    needs: pano.trunk_heading_deg
    paired_trunk_df: contained two trunks only
    '''

    ax.scatter(paired_trunk_df['pano_x'], paired_trunk_df['pano_y'],
               # legend=True,
               color='m')

    xmin, xmax, ymin, ymax = ax.axis()

    # Extend the bounds by extend meters
    ax.set_xlim([xmin - extend, xmax + extend])
    ax.set_ylim([ymin - extend, ymax + extend])

    for idx, row in paired_trunk_df.iterrows():

        # Draw pano yaw
        length = pano_len
        yaw = np.radians(90 - row['pano_heading_deg'])
        x_end = length * np.cos(yaw)
        y_end = length * np.sin(yaw)
        arrow = patches.Arrow(x=row['pano_x'], y=row['pano_y'], dx=x_end, dy=y_end, width=0.6, color="blue")
        ax.add_patch(arrow)


        ax.text(x=row['pano_x'] + 1, y=row['pano_y'] + 1, s=row['panoID'], c='blue')

        length = trunk_len
        yaw = np.radians(90 - row['trunk_heading_deg'])
        # draw trunk heading
        x_end = length * np.cos(yaw)
        y_end = length * np.sin(yaw)
        arrow = patches.Arrow(x=row['pano_x'], y=row['pano_y'], dx=x_end, dy=y_end, width=0.6, color="g")
        ax.add_patch(arrow)
        # heading_dif_deg = utils.degree_difference(pano.jdata['Projection']['pano_yaw_deg'], pano.trunk_heading_deg)
        # print(pano.panoId, "heading_dif_deg:", heading_dif_deg)



    # Adjust data limits to achieve 4:3 ratio (adjust based on actual data and desired behavior)
    x_lim_diff = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_lim_diff = ax.get_ylim()[1] - ax.get_ylim()[0]

    new_x_lim = [ax.get_xlim()[0], ax.get_xlim()[0] + (4 / 3) * y_lim_diff]
    ax.set_xlim(new_x_lim)

    return ax


def draw_yaw_arrow(ax, pano, length=3, width=0.3):
    '''
    length:
    needs: pano.trunk_heading_deg
    '''
    length = 3

    yaw = np.radians(90 - pano['pano_heading_deg'])

    # Draw pano yaw
    x_end = length * np.cos(yaw)
    y_end = length * np.sin(yaw)
    arrow = patches.Arrow(x=pano.pano_x, y=pano.pano_y, dx=x_end, dy=y_end, width=0.6, color="blue")
    ax.add_patch(arrow)

    ax.text(x=pano.pano_x + 1, y=pano.pano_y + 1, s=pano.panoID, c='blue')

    length = 20
    yaw = np.radians(90 - pano.trunk_heading_deg)

    # draw trunk heading
    x_end = length * np.cos(yaw)
    y_end = length * np.sin(yaw)
    arrow = patches.Arrow(x=pano.pano_x, y=pano.pano_y, dx=x_end, dy=y_end, width=0.6, color="g")
    ax.add_patch(arrow)
    # ax.axis('off')
    ax.set_aspect('equal')

    heading_dif_deg = utils.degree_difference(pano['pano_heading_deg'], pano.trunk_heading_deg)

def form_trunk_pairs(raw_df, near_thres=3, camera_dis_thres=20):
    '''
    near_thres: meter unit
    camera_dis_thres: meter
    '''
    df = raw_df.query(f" depth_h_distance < {camera_dis_thres} ")
    df = df.sort_values(['depth_x', 'depth_y'])
    pair_idx = []

    for i in range(len(df) - 1):
        diff_x = df.iloc[i]['depth_x'] - df.iloc[i + 1]['depth_x']
        diff_y = df.iloc[i]['depth_y'] - df.iloc[i + 1]['depth_y']
        distance = (diff_x ** 2 + diff_x ** 2) ** 0.5
        # print("distance: ", distance)
        if distance < near_thres:
            pair_idx.append(i)
            pair_idx.append(i + 1)

    return df.iloc[pair_idx]


def merge_trunk_df_pair(trunk_df, paired_trunk_df, local_crs):
    gdf_list = []
    for df in [trunk_df, paired_trunk_df]:
        # for trunk initial location from depthmap
        geometry = gpd.points_from_xy(df['depth_x'], df['depth_y'])
        gdf = gpd.GeoDataFrame(data=df, geometry=geometry).set_crs(local_crs)
        gdf['Type'] = 'trunk'
        gdf_list.append(gdf)

        # for pano (camera) location
        geometry = gpd.points_from_xy(df['pano_x'], df['pano_y'])
        gdf = gpd.GeoDataFrame(data=df.iloc[:1], geometry=geometry[:1]).set_crs(local_crs)
        gdf['Type'] = 'camera'
        gdf_list.append(gdf)

    all_gdf = pd.concat(gdf_list)
    return  all_gdf


def draw_trunk_depthmap_location(ax, all_gdf, extend=30, circle_size=3):
    # fig, ax = plt.subplots(figsize=(6, 6))
    category_colors = {'camera': 'green', 'trunk': 'm'}
    ListedColormap([category_colors[cat] for cat in all_gdf['Type'].unique()])


    cmap = ListedColormap([category_colors[cat] for cat in all_gdf['Type'].unique()])

    all_gdf.query("Type == 'camera' ").to_crs(epsg=3857).plot(ax=ax,
                                                              column='Type',
                                                              legend=False,
                                                              facecolor='none',
                                                              cmap=cmap)  # , marker_kwds={'radius':5}  # , column='Type', cmap='seismic'

    xmin, xmax, ymin, ymax = ax.axis()

    # # Extend the bounds by extend meters
    ax.set_xlim([xmin - extend, xmax + extend])
    ax.set_ylim([ymin - extend, ymax + extend])

    # draw trunk circles
    xmin, xmax, ymin, ymax = ax.axis()
    data_range_meters = max(xmax - xmin, ymax - ymin)
    marker_size_fraction = (circle_size / data_range_meters)
    dpi = plt.rcParams['figure.dpi']  # Get the current default DPI
    figsize = plt.rcParams['figure.figsize']  # Get the current default figure size
    conversion_factor = dpi * figsize[0]
    marker_size_points = (marker_size_fraction * conversion_factor) ** 2

    all_gdf_3857 = all_gdf.to_crs(epsg=3857)

    for idx, trunk in all_gdf_3857.query("Type == 'trunk' ").iterrows():
        # print("trunk:", trunk)
        # print("trunk.geometry.centroid:", trunk.geometry.centroid)
        (depth_x, depth_y) = trunk.geometry.centroid.xy
        scatter = ax.scatter(depth_x, depth_y, s=marker_size_points, facecolor='none', edgecolor='red')
        # circle = patches.Circle((depth_x, depth_y), 3, edgecolor='red',
        #                         facecolor='none',
        #                         label='trunk')
        # ax.add_patch(circle)

    # Create a custom legend
    legend_marker = mlines.Line2D([], [], color='red', marker='o', markersize=10,
                                  label='trunk (3m circle)',
                                  linestyle='None',
                                  markeredgewidth=0.8,
                                  markerfacecolor='none')
    camera_legend = mlines.Line2D([], [], color='blue', marker='o', markersize=8,
                                  label='camera',
                                  linestyle='None',
                                  markeredgewidth=0.0,
                                  markerfacecolor='m')

    # Add the custom legend to the plot
    ax.legend(handles=[camera_legend, legend_marker])

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)  # ,   source=ctx.providers.Stamen.Terrain



def draw_a_trunk_depthmap_location(ax, paired_trunk_df, local_crs, extend=30, circle_size=3):
    # fig, ax = plt.subplots(figsize=(6, 6))
    # print(paired_trunk_df)
    # print(paired_trunk_df[['pano_x', 'pano_y']],)
    camera_gdf = gpd.GeoDataFrame(data=paired_trunk_df,
                                  geometry=gpd.points_from_xy(paired_trunk_df['pano_x'],
                                                              paired_trunk_df['pano_y']),

                                  ).set_crs(local_crs)
    # ax.scatter(paired_trunk_df['pano_x'], paired_trunk_df['pano_y'],
    #            # legend=True,
    #            label='camera',
    #            color='m'
    #            )

    camera_gdf.to_crs(epsg=3857).plot(ax=ax, legend=False, color='m')  # for adding basemap

    xmin, xmax, ymin, ymax = ax.axis()

    # Extend the bounds by extend meters
    ax.set_xlim([xmin - extend, xmax + extend])
    ax.set_ylim([ymin - extend, ymax + extend])

    # Adjust data limits to achieve 4:3 ratio (adjust based on actual data and desired behavior)
    x_lim_diff = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_lim_diff = ax.get_ylim()[1] - ax.get_ylim()[0]

    new_x_lim = [ax.get_xlim()[0], ax.get_xlim()[0] + (4 / 3) * y_lim_diff]
    ax.set_xlim(new_x_lim)
    #
    # # draw trunk circles
    # xmin, xmax, ymin, ymax = ax.axis()
    data_range_meters = max(xmax - xmin, ymax - ymin)
    marker_size_fraction = (circle_size / data_range_meters)
    dpi = plt.rcParams['figure.dpi']  # Get the current default DPI
    figsize = plt.rcParams['figure.figsize']  # Get the current default figure size
    conversion_factor = dpi * figsize[0]
    marker_size_points = (marker_size_fraction * conversion_factor) ** 2

    depth_trunk_gdf = gpd.GeoDataFrame(data=paired_trunk_df,
                                  geometry=gpd.points_from_xy(paired_trunk_df['depth_x'],
                                                              paired_trunk_df['depth_y']),

                                  ).set_crs(local_crs).to_crs(epsg=3857) # for adding basemap

    for idx, trunk in depth_trunk_gdf.iterrows():
        # print("trunk:", trunk)
        # print("trunk.geometry.centroid:", trunk.geometry.centroid)
        (depth_x, depth_y) = trunk.geometry.centroid.xy
        # scatter = ax.scatter(depth_x, depth_y, s=marker_size_points, facecolor='none', edgecolor='red')
        circle = patches.Circle((depth_x, depth_y), 3, edgecolor='red',
                                facecolor='none',
                                label='trunk')
        ax.add_patch(circle)
    #
    # # Create a custom legend
    legend_marker = mlines.Line2D([], [], color='red', marker='o', markersize=10,
                                  label='trunk (3m radius circle)',
                                  linestyle='None',
                                  markeredgewidth=0.8,
                                  markerfacecolor='none')
    camera_legend = mlines.Line2D([], [], color='blue', marker='o', markersize=8,
                                  label='camera',
                                  linestyle='None',
                                  markeredgewidth=0.0,
                                  markerfacecolor='m')

    # Add the custom legend to the plot
    ax.legend(handles=[camera_legend, legend_marker])


    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)  # ,   source=ctx.providers.Stamen.Terrain


def triangulate_trunk_location(trunk_pair_df):
    '''
    trunk_pair_df: two-row dataframe for a pair of trunk
    '''
    # print("trunk_pair_df:", trunk_pair_df)
    pano0 = trunk_pair_df.iloc[0]
    pano1 = trunk_pair_df.iloc[1]
    pano_distance = utils.two_points_distance(
        x1=pano0['pano_x'],
        y1=pano0['pano_y'],
        x2=pano1['pano_x'],
        y2=pano1['pano_y'])


    # Form the triangle
    route_heading = utils.bearing_angle(pano0['pano_x'], pano0['pano_y'], pano1['pano_x'], pano1['pano_y'])
    # print("route_heading:", route_heading)

    angle_0 = utils.degree_difference(route_heading, pano0.trunk_heading_deg)

    route_heading = utils.bearing_angle(pano1['pano_x'], pano1['pano_y'], pano0['pano_x'], pano0['pano_y'])
    # print("route_heading:", route_heading)
    angle_1 = utils.degree_difference(route_heading, pano1.trunk_heading_deg)

    angle_2 = 180 - angle_1 - angle_0

    angle_2_rad = math.radians(angle_2)

    D = pano_distance / math.sin(angle_2_rad)

    dis_1 = D * math.sin(math.radians(angle_0))
    dis_0 = D * math.sin(math.radians(angle_1))

    trunk_pair_df['h_distance_to_camera'] = [dis_0, dis_1]
    trunk_pair_df['vertex_angle_deg'] = [angle_0, angle_1]

    # compute the horizontal location using triangulated result
    trunk_pair_df['pano_heading_rad'] = trunk_pair_df['pano_heading_deg'] / 180 * np.pi
    trunk_pair_df['tri_x'] = trunk_pair_df['pano_x'] + trunk_pair_df['h_distance_to_camera'] * np.cos(np.pi/2 - np.radians(trunk_pair_df['trunk_heading_deg']))
    trunk_pair_df['tri_y'] = trunk_pair_df['pano_y'] + trunk_pair_df['h_distance_to_camera'] * np.sin(np.pi/2 - np.radians(trunk_pair_df['trunk_heading_deg']))


    # print("pano_distance:", pano_distance)
    # print("angle_0:", angle_0)
    # print("angle_1:", angle_1)
    # print("angle_2:", angle_2)
    #
    # print("dis0, dis1:", dis_0, dis_1)

    return trunk_pair_df


def compute_diameter(trunk_pair_df, horizontal_fov_rad=math.radians(90), thumb_width=1024, thumb_height=768):

    trunk_pair_df['left_col'] = trunk_pair_df['root_center_col'] - trunk_pair_df['median_wid'] / 2
    trunk_pair_df['right_col'] = trunk_pair_df['root_center_col'] + trunk_pair_df['median_wid'] / 2
    angle_of_view_rad_list = []
    # get_angle_of_view
    for idx, row in trunk_pair_df.iterrows():
        phi_0, theat_0 = utils.row_col_to_angle(row=row['median_wid_row'],
                                                col=row['left_col'],
                                                width=thumb_width,
                                                height=thumb_height,
                                                horizontal_fov_rad=horizontal_fov_rad)


        phi_1, theat_1 = utils.row_col_to_angle(row=row['median_wid_row'],
                                                col=row['right_col'],
                                                width=thumb_width,
                                                height=thumb_height,
                                                horizontal_fov_rad=horizontal_fov_rad)

        angle_of_view_rad = abs(phi_1 - phi_0)
        angle_of_view_rad_list.append(angle_of_view_rad)

    trunk_pair_df['diameter'] = angle_of_view_rad_list
    trunk_pair_df['diameter'] = 2 * np.sin(trunk_pair_df['diameter'] / 2) * trunk_pair_df['h_distance_to_camera']

    return trunk_pair_df

def add_legend_to_diameter_measurement(ax):
    # # Create a custom legend
    diameter_marker = mlines.Line2D([], [], color='red',
                                  label='diameter',
                                  linestyle=(-5, (10, 20)),  # Adjust these numbers as needed
                                  linewidth=1,
                                  )

    diameter_marker.set_markersize(1)

    root_legend = mlines.Line2D([], [], color='blue', marker='o', markersize=5,
                                  label='tree root',
                                  linestyle='None',
                                  markeredgewidth=0.0,
                                  markerfacecolor='red')

    # Add the custom legend to the plot
    legend = ax.legend(handles=[diameter_marker, root_legend])

    # Create a legend and add the custom elements
    # legend = ax.legend(handles=[diameter_marker, root_legend])

    # Adjust the legend handle length (optional)
    legend.get_frame().set_linewidth(0.0)
    legend.get_title().set_fontsize(1)

    # Add the legend to the plot
    ax.add_artist(legend)


def show_detailed_images(trunk_pair_df, local_crs, image_dir, segmented_dir, save_dir):
    # Create a figure
    # print("trunk_pair_df:", trunk_pair_df)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 11.5))
    fig_fname = ''
    images = []
    basenames = []
    annotated_images = []
    image_pair = trunk_pair_df.iloc[0:2]['image_file'].to_list()
    # print("image_pair:", image_pair)

    for image_file in image_pair:
        try:
            # print("image_file:", image_file)
            basename = os.path.basename(image_file)
            basenames.append(basename)
            image_fname = os.path.join(image_dir, image_file)

            image_rgb = Image.open(image_fname)
            image_rgb = np.array(image_rgb)
            # image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # image = cv2.imread(image_fname)   # do not know why does not work for columbia images
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            images.append(image_rgb)

            annotated_fname = os.path.join(segmented_dir, basename.replace(".jpg", "_annotated.jpg"))

            # print("image_fname:", image_fname)
            # image_rgb = Image.open(image_fname)
            # image = cv2.imread(annotated_fname)  # do not know why does not work for columbia images
            image_rgb = Image.open(annotated_fname)
            image_rgb = np.array(image_rgb)
            # image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # print("annotated_fname:", annotated_fname)
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print("OK")
            annotated_images.append(image_rgb)

        except Exception as e:
            print("Error in loading image:", e, image_fname)

    mean_diameter = trunk_pair_df['diameter'].mean()

    for idx, image in enumerate(images):
        try:
            # print(idx)
            df = trunk_pair_df.iloc[idx:idx + 1]

            axs[0, idx].imshow(annotated_images[idx])
            # axs[0, idx].plot(df['root_center_col'], df['root_bottom_row'], 'o', color='red')
            # axs[0, idx].plot(df['root_center_col'], df['median_wid_row'], 'o', color='y')
            axs[0, idx].set_title(basenames[idx][:-4], y=1.03)
            axs[0, idx].axis('off')

            axs[1, idx].imshow(image)
            # draw the root dot
            axs[1, idx].plot(df['root_center_col'], df['root_bottom_row'], 'o',
                             color='red', markersize=3)
            # axs[1, idx].plot(df['root_center_col'], df['median_wid_row'], 'o', color='y')
            # print("image.shape:", image.shape)
            # print("df.iloc[0]['median_row_start_col']:", df.iloc[0]['median_row_start_col'])
            # print("df.iloc[0]['median_row_end_col']:", df.iloc[0]['median_row_end_col'])

            axs[1, idx].axhline(y=df.iloc[0]['median_wid_row'],
                                xmin=df.iloc[0]['median_row_start_col'] / image.shape[1],  # mage.shape: (768, 1024, 3)
                                xmax=df.iloc[0]['median_row_end_col'] / image.shape[1],
                                color='r', linestyle='-')
            axs[1, idx].axis('off')
            axs[1, idx].set_title(f"Diameter (m): {df.iloc[0]['diameter']:0.2f}, mean diameter: {mean_diameter:0.2f}")

            fig_fname = fig_fname + df.iloc[0]['panoID'] + "_" + str(df.iloc[0]['trunk_heading_deg'].astype(int)) + "_"



        except Exception as e:
            print("Error:", e)
            continue

    add_legend_to_diameter_measurement(axs[1, 1])

    draw_a_trunk_depthmap_location(ax=axs[2, 0],
                                       paired_trunk_df=trunk_pair_df,
                                       local_crs=local_crs,
                                       extend=30,
                                       circle_size=3)

    axs[2, 0].set_title("Trunk location from depthmap")

    draw_yaw_arrows(ax=axs[2, 1], paired_trunk_df=trunk_pair_df,
                        pano_len=3,
                        trunk_len=20,
                        width=0.3,
                        extend=20)
    axs[2, 1].set_title("Trunk location from triangulation")

    fig_fname = fig_fname + 'detailed_measurement.png'
    fig_fname = os.path.join(save_dir, fig_fname)
    plt.savefig(fig_fname, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
