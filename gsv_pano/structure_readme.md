# Street View Image + Metadata Structure

This note describes how `gsv_pano/pano.py` and `gsv_pano/utils.py` organize
Street View imagery, depth data, and metadata. It is written to help a new
reader understand the data products and how they relate.

## 1) Main objects and data flow

`GSV_pano` (class in `gsv_pano/pano.py`) is the center of the workflow:

- **Inputs**: `panoId`, or `(lon, lat)`, or a local `*.json` metadata file.
- **Metadata**: `getJsonfrmPanoID()` fetches raw Google photometa, then
  `utils.compressJson()` + `utils.refactorJson()` simplify it into a compact
  JSON schema (see Section 3).
- **Derived products**:
  - Panorama (equirectangular JPG assembled from tiles).
  - Depth map (from the compressed `model.depth_map` string in metadata).
  - Ground mask / normal vector map / plane index map (from depth parsing).
  - DEM and DOM (rasterized products with world files).

The data are cached in memory inside dictionaries on the `GSV_pano` instance
and optionally saved to disk under `saved_path`.

## 2) In-memory data organization

Each `GSV_pano` instance keeps a structured set of products:

- `self.panorama`
  - `image`: numpy array (RGB panorama, equirectangular).
  - `zoom`: zoom level used to fetch tiles.
- `self.depthmap`
  - `depthMap`: float depth array (meters, in the pano camera frame).
  - `dm_mask`: binary mask where depth is valid.
  - `ground_mask`: mask inferred from normal vectors.
  - `normal_vector_map`: per-pixel normal vectors (scaled to 0..255).
  - `plane_idx_map`: per-pixel plane indices from depth map parsing.
  - `width`, `height`, `zoom`: resolved image dimensions.
- `self.segmenation` (note: spelling in code)
  - `segmentation`: numpy array (palette or RGB segmentation).
  - `zoom`: assumed zoom for alignment.
  - `full_path`: file path set via `set_segmentation_path()`.
- `self.point_cloud`
  - `point_cloud`: xyz(+color/plane info) from depth map.
  - `zoom`, `dm_mask`.
- `self.DEM`
  - `DEM`: elevation raster.
  - `resolution`, `central_x`, `central_y`, `camera_height`, `zoom`.
- `self.DOM`
  - `DOM`: orthorectified image/segmentation raster.
  - `resolution`, `central_x`, `central_y`, `DOM_points`.

## 3) Metadata JSON schema (refactored)

`utils.refactorJson()` converts the raw Google response into a compact JSON
with top-level keys:

```
{
  "Data": {
    "image_width", "image_height",
    "tile_width", "tile_height",
    "level_sizes", "image_date",
    "imagery_type", "copyright"
  },
  "Projection": {
    "projection_type", "pano_yaw_deg",
    "tilt_yaw_deg", "tilt_pitch_deg"
  },
  "Location": {
    "panoId", "zoomLevels", "lat", "lng",
    "original_lat", "original_lng",
    "elevation_wgs84_m", "elevation_egm96_m",
    "description", "streetRange", "region", "country"
  },
  "Links": [
    {"panoId", "yawDeg", "road_argb", "description"}, ...
  ],
  "Time_machine": [
    {"panoId", "image_date", "lat", "lng", "elevation_egm96_m",
     "heading_deg", "yaw_deg", "pitch_deg", "description"}, ...
  ],
  "model": {
    "depth_map": "<compressed string>"
  }
}
```

Notes:
- `level_sizes` defines width/height per zoom for panoramas and depth maps.
- `Links` encodes neighboring panoramas with the outgoing yaw angle.
- `Time_machine` stores historical panoramas at the same location.
- `model.depth_map` is a compressed string that is decoded by
  `utils.parse()` -> `utils.parseHeader()` -> `utils.parsePlanes()` ->
  `utils.computeDepthMap()`.

## 4) Street view image structure

### 4.1 Panorama (equirectangular)

Panorama images are equirectangular mosaics of tiles:

- Tile fetch happens in `download_panorama()` using
  `https://geo{0-3}.ggpht.com/cbk?...&panoid=<panoId>&output=tile&x=<col>&y=<row>&zoom=<zoom>`.
- `tile_width`/`tile_height` and `level_sizes` come from JSON `Data`.
- The final panorama is cropped to the exact size from `level_sizes[zoom]`.

### 4.2 Depth map

Depth maps are derived from `model.depth_map` inside the JSON:

- `utils.parse()` decompresses the base64+zlib payload.
- `utils.parseHeader()` returns width/height/offset/plane count.
- `utils.parsePlanes()` reads plane normals and index map.
- `utils.computeDepthMap()` reconstructs a depth map (float).

The resulting `depthMap` is aligned to the panorama grid at the chosen zoom.

### 4.3 DEM (Digital Elevation Model)

`get_DEM()` and `calculate_DEM()` turn ground points from the depth map into a
local DEM grid. A world file is produced for GIS alignment.

### 4.4 DOM (Digital Orthoimage)

`get_DOM()` and `calculate_DOM()` drape panorama colors (or segmentation) onto
the DEM grid, producing an orthorectified raster and a matching world file.

## 5) File naming and saved outputs

When `saved_path` is provided, common outputs include:

- `panoId.json`
  - Refactored metadata JSON from `getJsonfrmPanoID()`.
- `panoId_<zoom>.jpg`
  - Panorama image at the given zoom from `get_panorama()`.
  - Prefix/suffix can be added in `get_panorama(prefix, suffix, ...)`.
- `panoId.tif`
  - Depth map saved by `get_depthmap(saved_path=...)` (single zoom used).
- `panoId_DEM_<resolution>.tif` + `panoId_DEM_<resolution>.tfw`
  - DEM raster + world file from `get_DEM()`.
- `panoId_DOM_<resolution>.<ext>` + matching world file
  - DOM raster + world file from `get_DOM()`.
  - The extension `ext` is inherited from the segmentation file when used.

Example: `aEfzJNuQkyR5KkeSG6mTRQ_DOM_0.05.tif` and
`aEfzJNuQkyR5KkeSG6mTRQ_DOM_0.05.tfw`.

## 6) How metadata and imagery connect

- **Georeferencing**: `Location.lat/lng` is the pano camera position.
  DEM/DOM world files use `central_x/central_y` from local coordinates derived
  from point cloud computation.
- **Orientation**: `Projection` fields drive pano orientation and the
  conversion between pano pixels and 3D directions.
- **Navigation**: `Links` and `Time_machine` enable traversing the pano graph
  and historical imagery.

## 7) Helpful code entry points

- `gsv_pano/pano.py`
  - `GSV_pano.__init__()`: constructs a pano object and loads metadata.
  - `getJsonfrmPanoID()`: fetches/refactors metadata.
  - `get_panorama()` / `download_panorama()`: image mosaics.
  - `get_depthmap()`: depth map and masks.
  - `get_DEM()` / `get_DOM()`: gridded GIS outputs.
- `gsv_pano/utils.py`
  - `refactorJson()`: compact JSON schema.
  - `parse()/parseHeader()/parsePlanes()/computeDepthMap()`: depth decoding.
