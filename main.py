# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:25:51 2023

@author: wjeanph
"""
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# we dont use it directly byt sqlalchemy needs it
import cx_Oracle
import geopandas as gpd
import keyring as kr
import pandas as pd
from pyproj import CRS
from shapely.geometry import MultiPolygon, Point, Polygon
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from setup_keyring import setup_keyring


########################################################################################################################
########################################################################################################################
def main():
    # PARAMETERS

    # Specify the database name, table name, UID, shape column name, spatial reference, and conditions for the query
    hostname = "Geodepot"
    database_name = "WAREHOUSE"
    table_name = "WC2021NGD_A_202106"
    uid = "BB_UID"
    shape_column = "SHAPE"
    spatial_reference = 3347
    # "PROJCS['c',GEOGCS['GCS_North_American_1983',DATUM['D_North_American_1983',SPHEROID['GRS_1980',6378137.0,298.257222101]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Lambert_Conformal_Conic'],PARAMETER['False_Easting',6200000.0],PARAMETER['False_Northing',3000000.0],PARAMETER['Central_Meridian',-91.86666666666666],PARAMETER['Standard_Parallel_1',49.0],PARAMETER['Standard_Parallel_2',77.0],PARAMETER['Latitude_Of_Origin',63.390675],UNIT['Meter',1.0]];3266364 256826 350;-100000 10000;-100000 10000;5.71428571428571E-03;0.001;0.001;IsHighPrecision"
    conditions = None  # Optional, set to None if not needed
    sample = False

    csv_long_lat_file = "C:/Users/wjeanph/Documents/python/point_in_poly/coordinates/poc_20230301_latlong_48b.csv"
    output_csv_file = "C:/Users/wjeanph/Documents/python/point_in_poly/coordinates/test_joined_poc_20230301_latlong_48b.csv"

    chunk_size = 100000  # Define your desired chunk size

    # Set to True to use parallel processing, False otherwise
    use_parallel = True

    # Set to True to use multi-threading, False to use multi-processing
    use_threads = False

    # Control the number of processes/threads
    max_workers = None  # None means it will use the default value

    # Controls whether we want to use a cached file instead of query the Oracle database
    use_bb_uid_cache = True

    ########################################################################################################################
    ########################################################################################################################
    target_crs = CRS.from_epsg(spatial_reference)

    # Specify the connection credentials
    cred = kr.get_credential(hostname, "")

    # print(username)
    # print(password)
    if cred:
        username = cred.username
        password = cred.password
    else:
        setup_keyring()

    # Specify the connection string
    connection_str = f"oracle+cx_oracle://{username}:{password}@{hostname}"

    # Call the function to query the database and get the GeoPandas DataFrame

    df_shapes, cached_used = get_df_shapes(
        use_bb_uid_cache,
        "bb_uid_shapes.pkl",
        connection_str,
        database_name,
        table_name,
        uid,
        shape_column,
        spatial_reference,
        conditions,
        sample,
    )

    print(f"Size of geopandas object returned: {get_object_size_in_gb(df_shapes)}")

    process_file_in_chunks(
        csv_long_lat_file,
        chunk_size,
        target_crs,
        df_shapes,
        output_csv_file,
        use_parallel,
        use_threads,
        max_workers,
    )


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        end_time = time.time()
        time_lapse = end_time - self.start_time
        time_lapse_minutes = time_lapse / 60.0
        print(f"Time lapse: {time_lapse_minutes:.2f} minutes")


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f"Execution time for {func.__name__}: {execution_time:.2f} minutes")
        return result

    return wrapper


def process_polygon_string(polygon_string):
    """Converts a polygon string to a Shapely MultiPolygon object.

    Args:
        polygon_string (str): The polygon string to process.

    Returns:
        Polygon or MultiPolygon: The processed Polygon or MultiPolygon object.
    """
    # Remove the outer "POLYGON ((" and "))" parts from the
    # print(type(polygon_string))
    # print(polygon_string)
    polygon_string = polygon_string.replace("POLYGON ((", "").replace("))", "")

    # Split the string into individual sub-polygon strings
    sub_polygon_strings = polygon_string.split("),(")

    # Convert each sub-polygon string to a Shapely Polygon object
    polygons = []
    for sub_polygon_string in sub_polygon_strings:
        # Split the sub-polygon string by commas and spaces to get individual coordinates
        coordinates = [
            tuple(map(float, coord.split())) for coord in sub_polygon_string.split(", ")
        ]
        # Create the Shapely Polygon object and add it to the list
        polygons.append(Polygon(coordinates))

    # Create a MultiPolygon object if there are multiple polygons, else use the single polygon
    if len(polygons) > 1:
        multipolygon = MultiPolygon(polygons)
    else:
        multipolygon = polygons[0]

    return multipolygon


def query_database_and_get_dataframe(
    connection_str,
    database_name,
    table_name,
    uid,
    shape_column,
    spatial_reference,
    conditions=None,
    sample=False,
):
    """Queries the database and retrieves a GeoPandas DataFrame.

    Args:
        connection_str (str): The Oracle database connection string.
        database_name (str): The name of the database.
        table_name (str): The name of the table.
        uid (str): The UID column name.
        shape_column (str): The shape column name.
        spatial_reference (str): The spatial reference system of the data.
        conditions (str, optional): Additional query conditions. Defaults to None.
        sample (bool, optional): Flag indicating whether to sample the data. Defaults to False.

    Returns:
        gpd.GeoDataFrame: The resulting GeoPandas DataFrame.
    """
    # Create a SQLAlchemy engine with connection pooling
    engine = create_engine(connection_str, poolclass=QueuePool)

    # Construct the SQL query
    query = f"SELECT {uid}, SDE.ST_asText({shape_column}) as GEOMETRY FROM {database_name}.{table_name}"

    if conditions or sample:
        query += " WHERE "

    if conditions:
        query += conditions

    if sample and conditions:
        query += " AND ROWNUM <= 100"
    elif sample and not conditions:
        query += "ROWNUM <= 100"

    timer = Timer()

    # Read the SQL query result into a DataFrame
    print("Querying database")
    timer.start()
    df = pd.read_sql(query, engine)
    timer.end()

    print("Converting shape to shapely geometry")
    timer.start()
    df["geometry"] = df["geometry"].apply(process_polygon_string)
    timer.end()

    gpdf = gpd.GeoDataFrame(df, geometry="geometry")

    gpdf.geometry.set_crs(crs=CRS.from_epsg(spatial_reference), inplace=True)

    return gpdf


# Function to handle cache checking and querying the database
@time_it
def get_df_shapes(
    use_bb_uid_cache,
    cache_path,
    connection_str,
    database_name,
    table_name,
    uid,
    shape_column,
    spatial_reference,
    conditions,
    sample,
):
    # Attempt to use cached file
    if use_bb_uid_cache:
        print("Attempting to use cached file for BB_UID data from Oracle:")
        if os.path.exists(cache_path):
            print(f"\t {cache_path} found and will be used!")
            # Load the GeoDataFrame from a pickle file
            with open(cache_path, "rb") as file:
                gdf = pickle.load(file)
            return gdf, True
        else:
            print(f"\t {cache_path} not found, will default to querying database!")

    # Not using cache or cache file not found
    gdf_shapes = query_database_and_get_dataframe(
        connection_str,
        database_name,
        table_name,
        uid,
        shape_column,
        spatial_reference,
        conditions,
        sample,
    )
    # Save the GeoDataFrame to a pickle file
    with open(cache_path, "wb") as file:
        pickle.dump(gdf_shapes, file)

    return gdf_shapes, False


# df_shapes.geometry.set_crs(spatial_reference, inplace =True)


def get_object_size_in_gb(obj):
    """
    Calculate the size of an object in gigabytes using pandas.

    Args:
        obj: The object for which to calculate the size.

    Returns:
        float: The size of the object in gigabytes.
    """
    # Get the memory usage of the object in bytes
    memory_usage = obj.memory_usage(deep=True).sum()

    # Convert the memory usage to gigabytes
    size_gb = memory_usage / (1024**3)

    return size_gb


def process_chunk(chunk, target_crs, df_shapes):
    geometry = [Point(xy) for xy in zip(chunk.longitude, chunk.latitude)]
    crs = CRS.from_epsg(4326)
    df_points = gpd.GeoDataFrame(chunk, geometry=geometry, crs=crs)

    print("Processing Chunk: Converting from epsg 4326 to epsg 3347")
    df_points.to_crs(target_crs, inplace=True)

    # Perform the spatial merge
    merged_gdf = gpd.sjoin(df_points, df_shapes, how="inner", predicate="within")
    return merged_gdf


@time_it
def process_file_in_chunks(
    csv_long_lat_file,
    chunk_size,
    target_crs,
    df_shapes,
    output_csv_file,
    use_parallel=False,
    use_threads=False,
    max_workers=None,
):
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    results = []

    with pd.read_csv(
        csv_long_lat_file,
        names=["unique_identifier", "latitude", "longitude"],
        chunksize=chunk_size,
    ) as reader:
        with executor_class(max_workers=max_workers) as executor:
            futures = []
            for chunk in reader:
                if use_parallel:
                    futures.append(
                        executor.submit(process_chunk, chunk, target_crs, df_shapes)
                    )
                else:
                    results.append(process_chunk(chunk, target_crs, df_shapes))

            if use_parallel:
                for future in futures:
                    results.append(future.result())

    merged_gdf = pd.concat(results, ignore_index=True)
    merged_gdf.to_csv(output_csv_file)
    print(merged_gdf.head())


if __name__ == "__main__":
    main()
