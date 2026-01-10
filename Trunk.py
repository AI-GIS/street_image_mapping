"""
Designed by Huan Ning, gladcolor@gmail.com, 2023.11.28
"""

# Python built-ins
import os
import time
import json
import csv
import math
from math import *
import sys
import struct
import base64
import zlib
import multiprocessing as mp
from skimage import morphology
from pyproj import CRS
from pyproj import Transformer

from sklearn.linear_model import LinearRegression
from PIL import ImageFilter
# Geospatial processing
from pyproj import Proj, transform
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from scipy.ndimage import gaussian_filter

import numpy as np
from numpy import inf

# import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from skimage import io
import yaml

import PIL
from PIL import Image, features
from PIL import Image, ImageDraw

import cv2

from io import BytesIO
import random

import requests
import urllib.request
import urllib
import logging


import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt


# import selenium
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
import sqlite3
from bs4 import BeautifulSoup
# WINDOWS_SIZE = '100, 100'
# chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
# Loading_time = 5

# import utils0
import utils

import logging.config


logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='Pano.log',
                filemode='w')


GROUND_VECTOR_THRES = 10
DEM_SMOOTH_SIGMA    = 1
DEM_COARSE_RESOLUTION = 0.8

def setup_logging(default_path='log_config.yaml', logName='', default_level=logging.INFO):
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if logName !='':
                config["handlers"]["file"]['filename'] = logName
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

yaml_path = 'log_config.yaml'
setup_logging(yaml_path)
logger = logging.getLogger('LOG.file')


logging.shutdown()

class Single_thumbnail_trunk(object):
    def __init__(self, image_fname, image, pano, depthmap, mask,  crs_local=None, saved_path=''):
        self.image_pairs = panoId  # test case: BM1Qt23drK3-yMWxYfOfVg
        self.pano = pano  # json object
        self.lat = None
        self.lon = None
        self.saved_path = saved_path
        self.crs_local = crs_local
        self.x = None
        self.y = None
        self.z = None
        self.normal_height = None


        # image storage: numpy array
        self.depthmap = depthmap
        self.panorama ={"image": None, "zoom": None}
        self.segmenation ={"segmentation": None, "zoom": None, 'full_path':None}
        self.point_cloud = {"point_cloud": None, "zoom": None, "dm_mask": None}
        self.DEM = {"DEM": None,
                    "zoom": None,
                    "resolution": None,
                    "elevation_wgs84_m": None,
                    "elevation_egm96_m": None,
                    "camera_height": None
                    }
        self.DOM = {"DOM": None, "zoom": None, "resolution": None,
                    "DOM_points": None} #

        try:

            if (self.panoId != 0) and (self.panoId is not None) and (len(str(self.panoId)) == 22):
                # print("panoid: ", self.panoId)
                self.jdata = self.getJsonfrmPanoID(panoId=self.panoId, dm=1, saved_path=self.saved_path)
                self.lon = self.jdata['Location']['lng']
                self.lat = self.jdata['Location']['lat']
            # else:
            #     logging.info("Found no paoraom in GSV_pano _init__(): %s" % panoId)

            if request_lat and request_lon:
                if (-180 <= request_lon <= 180) and (-90 <= request_lat <= 90):
                    self.panoId, self.lon, self.lat = self.getPanoIDfrmLonlat(request_lon, request_lat)



            if os.path.exists(self.json_file):
                try:
                    with open(self.json_file, 'r') as f:
                        jdata = json.load(f)
                        self.jdata = jdata
                        self.panoId = self.jdata['Location']['panoId']
                        self.lon = self.jdata['Location']['lng']
                        self.lat = self.jdata['Location']['lat']

                except Exception as e:
                    logging.info("Error in GSV_pano _init__() when loading local json file: %s, %s", self.json_file, e)
            # else:
            #     basename = os.path.basename(json_file)[:22]
            #     if panoId == 0:
            #         panoId = basename
            #         self.panoId = panoId







            # if self.crs_local and (self.lat is not None) and (self.lon is not None ):
            #     transformer = utils.epsg_transform(in_epsg=4326, out_epsg=self.crs_local)
            #     self.x, self.y = transformer.transform(self.lat, self.lon)



        except Exception as e:
            logging.exception("Error in GSV_pano _init__(): %s", e)



