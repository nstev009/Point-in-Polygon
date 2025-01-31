# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:25:51 2023

@author: wjeanph
"""
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

from pathlib import Path

# import cx_Oracle2 #called by sqlalchemy- explicitly added here to trigger warnings if not around
import geopandas as gpd
import keyring as kr
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from pyproj import CRS
from shapely.geometry import MultiPolygon, Point, Polygon
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from typing import Optional, Tuple

from setup_keyring import setup_keyring


########################################################################################################################
########################################################################################################################
class PointInPolygonConfig(BaseModel):
    # Required input/output files
    csv_long_lat_file: str = Field(..., description="Input CSV file with longitude/latitude coordinates")
    output_csv_file: str = Field(..., description="Output CSV file path")
    
    # Required database table information
    table_name: str = Field(..., description="Table name")
    
    # Database connection settings (with defaults)
    hostname: str = Field(default="Geodepot", description="Database hostname")
    database_name: str = Field(default="WAREHOUSE", description="Database name")
    
    # Database column and spatial settings
    uid: str = Field(default="BB_UID", description="UID column name for geodepot database")
    shape_column: str = Field(default="SHAPE", description="Shape column name for geodepot database")
    spatial_reference: int = Field(default=3347, description="Spatial reference system code for geodepot database")
    conditions: Optional[str] = Field(default=None, description="Additional query conditions for geodepot database")
    
    # CSV column mapping
    id_column: str = Field(default="rec_id", description="Name of the ID column in CSV")
    lat_column: str = Field(default="latitude", description="Name of the latitude column in CSV")
    lon_column: str = Field(default="longitude", description="Name of the longitude column in CSV")
    
    # Processing options
    chunk_size: int = Field(default=100000, description="Size of chunks for processing")
    use_parallel: bool = Field(default=True, description="Whether to use parallel processing")
    use_threads: bool = Field(default=False, description="Whether to use threads instead of processes")
    max_workers: Optional[int] = Field(default=None, description="Number of workers for parallel processing")
    return_all_points: bool = Field(default=True, description="If True, return all points with match status. If False, return only matched points.")
    match_status_column: str = Field(default="bb_uid_matched", description="Name of the column indicating whether a point matched a polygon")
    
    # Caching options
    use_bb_uid_cache: bool = Field(default=True, description="Whether to use cached BB_UID data")
    cache_dir: str = Field(default="cache", description="Directory to store cache files")
    cache_max_age_days: int = Field(default=120, description="Maximum age of cache files in days before they are considered stale")
    
    # Data sampling (for testing)
    sample: bool = Field(default=False, description="Whether to sample the data")

    @field_validator('csv_long_lat_file')
    def validate_csv_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"CSV file not found: {v}")
        return v
        
    def validate_csv_headers(self) -> None:
        """Validate that the CSV file has the required headers."""
        try:
            # Read just the header row
            headers = pd.read_csv(self.csv_long_lat_file, nrows=0).columns.tolist()
            
            required_columns = [self.id_column, self.lat_column, self.lon_column]
            
            missing_columns = [col for col in required_columns if col not in headers]
            if missing_columns:
                raise ValueError(
                    f"CSV file is missing required headers: {', '.join(missing_columns)}. "
                    f"Your CSV should have these columns: {', '.join(required_columns)}"
                )
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
            
    def __init__(self, **data):
        super().__init__(**data)
        self.validate_csv_headers()


def point_in_polygon(config: PointInPolygonConfig) -> None:
    """
    Process points to determine if they fall within polygons using the provided configuration.
    
    Args:
        config (PointInPolygonConfig): Configuration for the point-in-polygon operation
    """
    target_crs = CRS.from_epsg(config.spatial_reference)

    # Get credentials - use default username if not provided
    username = "wjeanph"  # default username
    password = kr.get_password(config.hostname, username)
    if not password:
        print("No credentials found in keyring. Running setup_keyring...")
        setup_keyring()
        # Try to get the password again after setup
        password = kr.get_password(config.hostname, username)
        if not password:
            print("Failed to set up credentials. Please run setup_keyring.py manually.")
            return

    # Specify the connection string
    connection_str = f"oracle+cx_oracle://{username}:{password}@{config.hostname}"

    # Query the database and get the GeoPandas DataFrame
    df_shapes, cached_used = get_df_shapes(
        config.use_bb_uid_cache,
        config.cache_dir,
        connection_str,
        config.database_name,
        config.table_name,
        config.uid,
        config.shape_column,
        config.spatial_reference,
        config.conditions,
        config.sample,
        config.cache_max_age_days,
    )

    print(f"Size of geopandas object returned: {get_object_size_in_gb(df_shapes)}")

    process_file_in_chunks(
        config.csv_long_lat_file,
        config.chunk_size,
        target_crs,
        df_shapes,
        config.output_csv_file,
        config.use_parallel,
        config.use_threads,
        config.max_workers,
        config.id_column,
        config.lat_column,
        config.lon_column,
        config.return_all_points,
        config.match_status_column,
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

    # Construct the SQL query with schema
    query = f"SELECT {uid}, SDE.ST_asText({shape_column}) as geometry FROM {database_name}.{table_name}"

    where_conditions = []
    if conditions:
        where_conditions.append(conditions)
    if sample:
        where_conditions.append("ROWNUM <= 100")

    if where_conditions:
        query += " WHERE " + " AND ".join(where_conditions)

    timer = Timer()

    try:
        # Read the SQL query result into a DataFrame
        print(f"Querying database: {query}")
        timer.start()
        df = pd.read_sql(query, engine)
        timer.end()

        if df.empty:
            print("Warning: Query returned no results")
            return gpd.GeoDataFrame()

        # Print column names for debugging
        print("Available columns:", df.columns.tolist())

        print("Converting shape to shapely geometry")
        timer.start()
        # Convert column names to lowercase for case-insensitive matching
        df.columns = df.columns.str.lower()
        df["geometry"] = df["geometry"].apply(process_polygon_string)
        timer.end()

        gpdf = gpd.GeoDataFrame(df, geometry="geometry")
        gpdf.geometry.set_crs(crs=CRS.from_epsg(spatial_reference), inplace=True)

        return gpdf
        
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        raise


# Function to handle cache checking and querying the database
@time_it
def get_df_shapes(
    use_bb_uid_cache: bool,
    cache_dir: str,
    connection_str: str,
    database_name: str,
    table_name: str,
    uid: str,
    shape_column: str,
    spatial_reference: int,
    conditions: Optional[str],
    sample: bool,
    max_age_days: int = 120,
) -> Tuple[gpd.GeoDataFrame, bool]:
    """
    Get shapes DataFrame, either from cache or by querying the database.
    
    Args:
        use_bb_uid_cache: Whether to use cached data
        cache_dir: Directory to store cache files
        connection_str: Database connection string
        database_name: Name of the database
        table_name: Name of the table
        uid: UID column name
        shape_column: Shape column name
        spatial_reference: Spatial reference system code
        conditions: Additional query conditions
        sample: Whether to sample the data
        max_age_days: Maximum age of cache files in days
        
    Returns:
        Tuple[gpd.GeoDataFrame, bool]: The shapes DataFrame and whether cache was used
    """
    if use_bb_uid_cache:
        # Clean up old cache files
        cleanup_old_caches(cache_dir, table_name, max_age_days)
        
        # Try to find a valid cache file
        cache_path = find_latest_cache(cache_dir, table_name, max_age_days)
        if cache_path and cache_path.exists():
            try:
                with cache_path.open("rb") as file:
                    gdf_shapes = pickle.load(file)
                print(f"Using cached shapes from: {cache_path}")
                return gdf_shapes, True
            except Exception as e:
                print(f"Error reading cache file: {e}")

    # If no cache or cache is invalid, query the database
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

    if use_bb_uid_cache:
        # Save to a new cache file with timestamp
        cache_path = get_cache_path(cache_dir, table_name)
        try:
            with cache_path.open("wb") as file:
                pickle.dump(gdf_shapes, file)
            print(f"Saved shapes to cache: {cache_path}")
        except Exception as e:
            print(f"Error saving cache file: {e}")

    return gdf_shapes, False


def get_cache_path(cache_dir: str, table_name: str) -> Path:
    """
    Generate a cache file path based on table name.
    The file name will include the table name and creation timestamp.
    
    Args:
        cache_dir: Directory to store cache files
        table_name: Name of the table being cached
        
    Returns:
        Path: Path to the cache file
    """
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Clean table name to be filesystem-friendly
    safe_table_name = "".join(c if c.isalnum() else "_" for c in table_name)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return cache_path / f"{safe_table_name}_{timestamp}.pkl"


def find_latest_cache(cache_dir: str, table_name: str, max_age_days: int) -> Optional[Path]:
    """
    Find the latest valid cache file for a given table.
    
    Args:
        cache_dir: Directory containing cache files
        table_name: Name of the table to find cache for
        max_age_days: Maximum age of cache files in days
        
    Returns:
        Optional[Path]: Path to the latest valid cache file, or None if no valid cache exists
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None
        
    # Clean table name to be filesystem-friendly
    safe_table_name = "".join(c if c.isalnum() else "_" for c in table_name)
    
    # Find all cache files for this table
    cache_files = []
    for file in cache_path.glob(f"{safe_table_name}_*.pkl"):
        file_time = datetime.fromtimestamp(file.stat().st_mtime)
        age_days = (datetime.now() - file_time).total_seconds() / (24 * 3600)
        
        if age_days <= max_age_days:
            cache_files.append((file, file_time))
    
    if not cache_files:
        return None
    
    # Return the most recent cache file
    latest_cache = max(cache_files, key=lambda x: x[1])[0]
    return latest_cache


def cleanup_old_caches(cache_dir: str, table_name: str, max_age_days: int) -> None:
    """
    Remove old cache files for a given table.
    
    Args:
        cache_dir: Directory containing cache files
        table_name: Name of the table to clean caches for
        max_age_days: Maximum age of cache files in days
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return
        
    # Clean table name to be filesystem-friendly
    safe_table_name = "".join(c if c.isalnum() else "_" for c in table_name)
    
    current_time = datetime.now()
    for file in cache_path.glob(f"{safe_table_name}_*.pkl"):
        file_time = datetime.fromtimestamp(file.stat().st_mtime)
        age_days = (current_time - file_time).total_seconds() / (24 * 3600)
        
        if age_days > max_age_days:
            try:
                file.unlink()
                print(f"Removed old cache file: {file.name}")
            except OSError as e:
                print(f"Error removing old cache file {file.name}: {e}")


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


def process_chunk(chunk, target_crs, df_shapes, lon_column="longitude", lat_column="latitude", return_all_points=False, match_status_column="bb_uid_matched"):
    geometry = [Point(xy) for xy in zip(chunk[lon_column], chunk[lat_column])]
    crs = CRS.from_epsg(4326)
    df_points = gpd.GeoDataFrame(chunk, geometry=geometry, crs=crs)

    print("Processing Chunk: Converting from epsg 4326 to epsg 3347")
    df_points.to_crs(target_crs, inplace=True)

    if return_all_points:
        # Perform left join to keep all points
        merged_gdf = gpd.sjoin(df_points, df_shapes, how="left", predicate="within")
        # Get the name of the index column from the right DataFrame after the join
        # This will be either 'index_right' or the actual index name if it exists
        index_right_col = 'index_right' if df_shapes.index.name is None else df_shapes.index.name
        # Add a match status column (True if point matched a polygon, False if not)
        merged_gdf[match_status_column] = ~merged_gdf[index_right_col].isna()
        # Drop the index_right column as it's not needed
        merged_gdf = merged_gdf.drop(columns=[index_right_col])
    else:
        # Original behavior: only return matched points
        merged_gdf = gpd.sjoin(df_points, df_shapes, how="inner", predicate="within")
        if df_shapes.index.name is not None:
            merged_gdf = merged_gdf.drop(columns=[df_shapes.index.name])
        else:
            merged_gdf = merged_gdf.drop(columns=['index_right'])

    return merged_gdf


@time_it
def process_file_in_chunks(
    csv_long_lat_file: str,
    chunk_size: int,
    target_crs: CRS,
    df_shapes: gpd.GeoDataFrame,
    output_csv_file: str,
    use_parallel: bool = False,
    use_threads: bool = False,
    max_workers: Optional[int] = None,
    id_column: str = "rec_id",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    return_all_points: bool = False,
    match_status_column: str = "bb_uid_matched",
) -> None:
    """
    Process a CSV file in chunks, performing spatial joins with the shapes DataFrame.
    
    Args:
        csv_long_lat_file: Path to input CSV file (must have headers)
        chunk_size: Number of rows to process at once
        target_crs: Target coordinate reference system
        df_shapes: GeoDataFrame containing polygon shapes
        output_csv_file: Path to output CSV file
        use_parallel: Whether to use parallel processing
        use_threads: Whether to use threads instead of processes
        max_workers: Number of worker threads/processes
        id_column: Name of the ID column in CSV
        lat_column: Name of the latitude column in CSV
        lon_column: Name of the longitude column in CSV
        return_all_points: If True, return all points with match status. If False, return only matched points.
        match_status_column: Name of the column indicating whether a point matched a polygon
    """
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    results = []

    # Verify required columns are present
    headers = pd.read_csv(csv_long_lat_file, nrows=0).columns.tolist()
    required_columns = [id_column, lat_column, lon_column]
    missing_columns = [col for col in required_columns if col not in headers]
    if missing_columns:
        raise ValueError(
            f"CSV file is missing required headers: {', '.join(missing_columns)}. "
            f"Expected headers: {', '.join(required_columns)}"
        )

    with pd.read_csv(
        csv_long_lat_file,
        chunksize=chunk_size,
        usecols=required_columns,  # Only read the columns we need
    ) as reader:
        with executor_class(max_workers=max_workers) as executor:
            futures = []
            for chunk in reader:
                if use_parallel:
                    futures.append(
                        executor.submit(process_chunk, chunk, target_crs, df_shapes, lon_column, lat_column, return_all_points, match_status_column)
                    )
                else:
                    results.append(process_chunk(chunk, target_crs, df_shapes, lon_column, lat_column, return_all_points, match_status_column))

            if use_parallel:
                for future in futures:
                    results.append(future.result())

    merged_gdf = pd.concat(results, ignore_index=True)
    merged_gdf.to_csv(output_csv_file, index=False)
    print(merged_gdf.head())


def main():
    # Example configuration with default values
    config = PointInPolygonConfig(
        csv_long_lat_file=r"\\fld6filer\Record_Linkage\Point_in_Polygon_Examples\input_test_data\bg_latlongs_100_recs_with_header.csv",
        output_csv_file="output_results.csv",  # Output in current directory
        table_name="WC2021NGD_A_202106",
        hostname="Geodepot",
        database_name="WAREHOUSE",
        uid="BB_UID",
        shape_column="SHAPE",
        spatial_reference=3347,
        id_column="bg_sn",  # Updated to match input file
        lat_column="bg_latitude",  # Updated to match input file
        lon_column="bg_longitude",  # Updated to match input file
        return_all_points=True
    )
    
    point_in_polygon(config)

if __name__ == "__main__":
    main()
