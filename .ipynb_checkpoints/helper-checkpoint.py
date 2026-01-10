import geopandas as gpd


# find geometry intersections within the same dataset using geopandas
def find_intersections(gdf: gpd.GeoDataFrame):
    # Save geometries to another field
    gdf['geom'] = gdf.geometry

    # Self join
    sj = gpd.sjoin(gdf, gdf,
                   how="inner",
                   predicate="intersects",
                   lsuffix="left",
                   rsuffix="right")

    # Remove geometries that intersect themselves
    sj = sj[sj.index != sj.index_right]

    # Extract the intersecting geometry
    sj['intersection_geom'] = sj['geom_left'].intersection(sj['geom_right'])

    # Reset the geometry (remember to set the CRS correctly!)
    sj.set_geometry('intersection_geom', drop=True, inplace=True, crs=gdf.crs)

    # Drop duplicate geometries
    final_gdf = sj.drop_duplicates(subset=['geometry']).reset_index()

    # Drop intermediate fields
    drops = ['geom_left', 'geom_right', 'index_right', 'index']
    final_gdf = final_gdf.drop(drops, axis=1)

    return final_gdf




