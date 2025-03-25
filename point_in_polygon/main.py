# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:25:51 2023

@author: wjeanph
"""
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import multiprocessing
from sqlalchemy import text

# import cx_Oracle2 #called by sqlalchemy- explicitly added here to trigger warnings if not around
import geopandas as gpd
import keyring as kr
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
from pyproj import CRS
from shapely.geometry import MultiPolygon, Point, Polygon
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from typing import Optional, Tuple
import os

from point_in_polygon.setup_keyring import setup_keyring

import os
# Set Oracle environment variables at the start of your script
os.environ['ORACLE_HOME'] = r"C:\ora19c\product\19.0.0\client_2"
os.environ['PATH'] = os.environ.get('PATH', '') + r";C:\ora19c\product\19.0.0\client_2\bin"
os.environ['TNS_ADMIN'] = r"\\stpfsmora01sa.file.core.windows.net\tnsnames\ora19"


########################################################################################################################
########################################################################################################################
class PointInPolygonConfig(BaseModel):
    # Required input/output files
    csv_long_lat_file: str = Field(..., description="Input CSV file with longitude/latitude coordinates")
    output_csv_file: str = Field(..., description="Output CSV file path")
    
    # Required database table information
    table_name: str = Field(..., description="Table name")
    
    # Database connection settings (with defaults)
    hostname: str = Field(default="geodepot", description="Database hostname")
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
    max_workers: Optional[int] = Field(default=None, description="Number of workers for parallel processing. If None, uses all available CPU cores")
    return_all_points: bool = Field(default=True, description="If True, return all points with match status. If False, return only matched points.")
    match_status_column: str = Field(default="bb_uid_matched", description="Name of the column indicating whether a point matched a polygon")
    
    # Caching options
    use_bb_uid_cache: bool = Field(default=True, description="Whether to use cached BB_UID data")
    cache_dir: str = Field(default="cache", description="Directory to store cache files")
    cache_max_age_days: int = Field(default=120, description="Maximum age of cache files in days before they are considered stale")
    
    # Data sampling (for testing)
    sample: bool = Field(default=False, description="Whether to sample the data")

    def __init__(self, *args, **kwargs):
        # Perform system checks automatically
        from point_in_polygon.utils import check_oracle_client, list_tables
        
        if not check_oracle_client():
            raise RuntimeError("Oracle client check failed")
        
        # Create cache directory if it doesn't exist
        os.makedirs(kwargs.get('cache_dir', 'cache'), exist_ok=True)
        
        # Check if credentials exist in keyring
        hostname = kwargs.get('hostname', 'Geodepot')
        service_name = get_service_name(hostname)  # Use get_service_name for consistency
        cred = kr.get_credential(service_name, "")
        if not cred:
            print(f"No credentials found for {hostname}. Setting up now...")
            setup_keyring(hostname)
            cred = kr.get_credential(hostname, "")
            if not cred:
                raise RuntimeError("Failed to set up credentials. Please try again.")
        
        super().__init__(*args, **kwargs)
        self.validate_csv_headers()
        
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
            
########################################################################################################################
########################################################################################################################
def get_service_name(hostname: str) -> str:
    """Get the service name for keyring storage."""
    return f"point_in_polygon_{hostname.lower()}"

def get_credentials(hostname: str) -> Tuple[str, str]:
    """Get database credentials from keyring."""
    service_name = get_service_name(hostname)
    
    # Try to get username from keyring first
    username = kr.get_password(service_name, "username")
    if not username:
        print("No username found in keyring. Running setup_keyring...")
        setup_keyring()
        username = kr.get_password(service_name, "username")
        if not username:
            raise ValueError("Failed to get username from keyring. Please run setup_keyring.py manually.")
    
    # Get password for the username
    password = kr.get_password(service_name, "password")
    if not password:
        print("No password found in keyring. Running setup_keyring...")
        setup_keyring()
        password = kr.get_password(service_name, "password")
        if not password:
            raise ValueError("Failed to get password from keyring. Please run setup_keyring.py manually.")
    
    return username, password


def point_in_polygon(config: PointInPolygonConfig) -> None:
    """
    Process points to determine if they fall within polygons using the provided configuration.
    
    Args:
        config (PointInPolygonConfig): Configuration for the point-in-polygon operation
    """
    target_crs = CRS.from_epsg(config.spatial_reference)

    try:
        # Get credentials from keyring
        username, password = get_credentials(config.hostname)
        
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
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        return


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        end_time = time.time()
        time_lapse = end_time - self.start_time
        time_lapse_minutes = time_lapse / 60.0
        return time_lapse_minutes


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f"Execution time for {func.__name__}: {execution_time:.2f} minutes")
        return result

    return wrapper


def process_chunk(chunk, target_crs, df_shapes, lon_column="longitude", lat_column="latitude", return_all_points=False, match_status_column="bb_uid_matched"):
    try:
        # Process the chunk without detailed logging
        geometry = [Point(xy) for xy in zip(chunk[lon_column], chunk[lat_column])]
        crs = CRS.from_epsg(4326)
        df_points = gpd.GeoDataFrame(chunk, geometry=geometry, crs=crs)

        # Convert coordinates silently
        df_points.to_crs(target_crs, inplace=True)

        if return_all_points:
            merged_gdf = gpd.sjoin(df_points, df_shapes, how="left", predicate="within")
            index_right_col = 'index_right' if df_shapes.index.name is None else df_shapes.index.name
            merged_gdf[match_status_column] = ~merged_gdf[index_right_col].isna()
            merged_gdf = merged_gdf.drop(columns=[index_right_col])
        else:
            merged_gdf = gpd.sjoin(df_points, df_shapes, how="inner", predicate="within")

        return merged_gdf
    except Exception as e:
        print(f"\nError processing points: {str(e)}")
        raise


def query_database_and_get_dataframe(
    connection_str,
    database_name,
    table_name,
    uid,
    shape_column,
    spatial_reference,
    conditions=None,
    sample=False,
    num_chunks=16,      # Number of chunks to split the data into
    max_workers=8       # Number of parallel workers
):
    """Queries the database in parallel chunks based on ID ranges and retrieves a GeoPandas DataFrame."""
    print("\n=== Querying Database ===")
    
    # Create a SQLAlchemy engine with connection pooling
    engine = create_engine(connection_str, poolclass=QueuePool)

    # Base query template with ID range parameters
    query_template = f"""
    SELECT {uid} as uid_col, SDE.ST_asText({shape_column}) as geometry 
    FROM {database_name}.{table_name}
    WHERE {uid} >= {{start_id}} AND {uid} < {{end_id}}
    """

    # Add additional conditions if provided
    if conditions:
        query_template += f" AND ({conditions})"
    if sample:
        query_template += " AND ROWNUM <= 100"

    print("• Getting table information...")
    chunk_boundaries, total_rows = get_chunk_boundaries(
        engine, database_name, table_name, uid, num_chunks, conditions
    )
    
    if not chunk_boundaries:
        print("Warning: No data found in specified range")
        return gpd.GeoDataFrame()

    print(f"• Found {total_rows:,} rows")
    print(f"• Using {max_workers} parallel workers to fetch {len(chunk_boundaries)} chunks")
    
    # Prepare arguments for parallel processing
    chunk_args = [
        (connection_str, query_template, start_id, end_id, i+1)
        for i, (start_id, end_id) in enumerate(chunk_boundaries)
    ]

    timer = Timer()
    timer.start()
    
    # Create progress bar for database fetching
    with tqdm(total=len(chunk_boundaries), desc="• Fetching data", unit="chunk", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]') as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_chunk_by_id_range, args) for args in chunk_args]
            
            dfs = []
            for future in as_completed(futures):
                chunk_df = future.result()
                if chunk_df is not None and not chunk_df.empty:
                    chunk_df.columns = chunk_df.columns.str.lower()
                    dfs.append(chunk_df)
                pbar.update(1)
    
    if not dfs:
        print("\nWarning: No data was fetched successfully")
        return gpd.GeoDataFrame()

    print("\n• Combining results...")
    df = pd.concat(dfs, ignore_index=True)
    rows_fetched = len(df)
    print(f"• Successfully fetched {rows_fetched:,} rows ({rows_fetched/total_rows:.1%} of total)")

    print("• Converting geometries...")
    geometry = df['geometry'].apply(process_polygon_string)
    
    # Create GeoDataFrame with the normalized column name
    gdf = gpd.GeoDataFrame(df[['uid_col']], geometry=geometry, crs=f"EPSG:{spatial_reference}")
    gdf = gdf.rename(columns={'uid_col': uid})
    
    elapsed_time = timer.end()
    print(f"\n=== Database query completed in {elapsed_time:.1f} minutes ===\n")

    return gdf


def fetch_chunk_by_id_range(args):
    """Fetch a chunk of data from the database using ID range."""
    connection_str, query_template, start_id, end_id, chunk_num = args
    engine = create_engine(connection_str)
    
    try:
        with engine.connect() as connection:
            chunk_df = pd.read_sql(text(query_template.format(start_id=start_id, end_id=end_id)), connection)
            return chunk_df
    except Exception as e:
        print(f"\nError in chunk {chunk_num}: {str(e)}")
        return None


def get_total_rows(engine, database_name, table_name, conditions=None):
    """Get total number of rows in the table."""
    query = f"SELECT COUNT(*) as count FROM {database_name}.{table_name}"
    if conditions:
        query += f" WHERE {conditions}"
    
    with engine.connect() as connection:
        result = connection.execute(text(query)).fetchone()
        return result[0]


def get_chunk_boundaries(engine, database_name, table_name, uid, num_chunks, conditions=None):
    """Get the boundaries for each chunk based on the UID values."""
    # Get min and max UID values
    query = f"""
    SELECT MIN({uid}) as min_id, MAX({uid}) as max_id, COUNT(*) as total_rows 
    FROM {database_name}.{table_name}
    """
    if conditions:
        query += f" WHERE {conditions}"
    
    with engine.connect() as connection:
        result = connection.execute(text(query)).fetchone()
        min_id, max_id, total_rows = result
        
    if not min_id or not max_id:
        return [], 0
        
    # Calculate chunk boundaries
    id_range = max_id - min_id
    chunk_size = id_range // num_chunks
    
    boundaries = []
    for i in range(num_chunks):
        start_id = min_id + (i * chunk_size)
        end_id = min_id + ((i + 1) * chunk_size) if i < num_chunks - 1 else max_id + 1
        boundaries.append((start_id, end_id))
        
    return boundaries, total_rows


def split_dataframe(df, n_chunks):
    """Split a DataFrame into n roughly equal chunks without using numpy's deprecated swapaxes."""
    chunk_size = len(df) // n_chunks
    remainder = len(df) % n_chunks
    chunks = []
    start = 0
    
    for i in range(n_chunks):
        # Add one extra row for chunks that should handle the remainder
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        chunks.append(df.iloc[start:end].copy())
        start = end
    
    return chunks


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
):
    """Process a CSV file in chunks, performing spatial joins with the shapes DataFrame."""
    print("\n=== Processing Points ===")
    
    # Get total number of chunks for progress bar
    total_chunks = sum(1 for _ in pd.read_csv(csv_long_lat_file, chunksize=chunk_size))
    
    if use_parallel:
        if max_workers is None:
            # Use all available CPU cores if max_workers is not specified
            max_workers = multiprocessing.cpu_count()
        else:
            # Ensure max_workers doesn't exceed the number of available cores
            max_workers = min(max_workers, multiprocessing.cpu_count())
        print(f"• Using {max_workers} parallel workers for {total_chunks} chunks")
    
    # Create progress bar for chunks
    with tqdm(total=total_chunks, desc="• Processing", unit="chunk", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]') as pbar:
        # Initialize the output file with headers
        first_chunk = True
        
        # Process the file in chunks
        if use_parallel:
            # Use ProcessPoolExecutor for true parallelism
            executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
            with executor_class(max_workers=max_workers) as executor:
                # Create a queue of futures
                futures = {}
                completed_order = []
                chunk_buffer = {}
                next_chunk_to_write = 1
                
                # Submit initial batch of chunks
                for chunk_num, chunk in enumerate(pd.read_csv(csv_long_lat_file, chunksize=chunk_size), 1):
                    # Split chunk for parallel processing
                    sub_chunks = split_dataframe(chunk, max_workers)
                    chunk_results = []
                    
                    # Submit all sub-chunks
                    for sub_chunk in sub_chunks:
                        future = executor.submit(
                            process_chunk,
                            sub_chunk,
                            target_crs,
                            df_shapes,
                            lon_column,
                            lat_column,
                            return_all_points,
                            match_status_column,
                        )
                        futures[future] = (chunk_num, len(chunk_results))
                        chunk_results.append(None)
                    
                    # Store the placeholder for results
                    chunk_buffer[chunk_num] = chunk_results
                    
                    # Process completed futures as they finish
                    while futures:
                        # Wait for the next future to complete
                        done, _ = concurrent.futures.wait(
                            futures.keys(), 
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        for future in done:
                            chunk_num, sub_chunk_idx = futures[future]
                            try:
                                result = future.result()
                                if result is not None:
                                    chunk_buffer[chunk_num][sub_chunk_idx] = result
                            except Exception as e:
                                print(f"\nError in chunk {chunk_num}: {str(e)}")
                            
                            # Remove the completed future
                            del futures[future]
                            
                            # Check if this chunk is complete
                            if chunk_num == next_chunk_to_write and all(r is not None for r in chunk_buffer[chunk_num]):
                                # All sub-chunks for this chunk are done, concatenate and write
                                try:
                                    results = [r for r in chunk_buffer[chunk_num] if r is not None]
                                    if results:
                                        result_chunk = pd.concat(results, ignore_index=True)
                                        mode = "w" if first_chunk else "a"
                                        header = first_chunk
                                        result_chunk.to_csv(output_csv_file, mode=mode, header=header, index=False)
                                        first_chunk = False
                                except Exception as e:
                                    print(f"\nError writing chunk {chunk_num}: {str(e)}")
                                
                                # Clean up the buffer and move to next chunk
                                del chunk_buffer[chunk_num]
                                next_chunk_to_write += 1
                                pbar.update(1)
        else:
            # Non-parallel processing
            for chunk_num, chunk in enumerate(pd.read_csv(csv_long_lat_file, chunksize=chunk_size), 1):
                try:
                    result_chunk = process_chunk(
                        chunk,
                        target_crs,
                        df_shapes,
                        lon_column,
                        lat_column,
                        return_all_points,
                        match_status_column,
                    )
                    
                    if result_chunk is not None:
                        mode = "w" if first_chunk else "a"
                        header = first_chunk
                        result_chunk.to_csv(output_csv_file, mode=mode, header=header, index=False)
                        first_chunk = False
                except Exception as e:
                    print(f"\nError in chunk {chunk_num}: {str(e)}")
                    continue
                
                pbar.update(1)
    
    print("\n=== Processing completed ===\n")


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


def main():
    # Example configuration with optimized values for small datasets
    config = PointInPolygonConfig(
        csv_long_lat_file=r"\\fld6filer\Record_Linkage\Point_in_Polygon_Examples\TEST_AREA_FOR_EVERYONE\data_input\my_latlongs_100_recs_sans_blanks_sans_oos.csv",
        output_csv_file="output_results.csv",  # Output in current directory
        table_name="WC2021NGD_A_202106",
        hostname="Geodepot",
        database_name="WAREHOUSE",
        uid="BB_UID",
        shape_column="SHAPE",
        spatial_reference=3347,
        id_column="my_uid",  # Updated to match input file
        lat_column="my_latitude",  # Updated to match input file
        lon_column="my_longitude",  # Updated to match input file
        return_all_points=True,
        chunk_size=100_000,  # Process in smaller chunks for better progress visibility
        use_parallel=True,  # Keep parallel processing on
        max_workers=4  # Limit workers for small dataset
    )
    
    point_in_polygon(config)


def check_oracle_config():
    """Check Oracle configuration and print diagnostic information."""
    import os
    import sys
    
    print("\n=== Oracle Configuration ===")
    print(f"ORACLE_HOME: {os.environ.get('ORACLE_HOME')}")
    print(f"TNS_ADMIN: {os.environ.get('TNS_ADMIN')}")
    
    # Check if Oracle bin directory is in PATH
    oracle_home = os.environ.get('ORACLE_HOME')
    oracle_bin = os.path.join(oracle_home, 'bin') if oracle_home else None
    path_env = os.environ.get('PATH', '')
    
    print(f"Oracle home in PATH: {oracle_home in path_env}")
    print(f"Oracle bin in PATH: {oracle_bin in path_env if oracle_bin else False}")
    
    # Check if TNS_ADMIN directory exists and contains tnsnames.ora
    tns_admin = os.environ.get('TNS_ADMIN')
    if tns_admin and os.path.isdir(tns_admin):
        print(f"TNS_ADMIN directory exists: {tns_admin}")
        tnsnames_path = os.path.join(tns_admin, 'tnsnames.ora')
        if os.path.isfile(tnsnames_path):
            print(f"tnsnames.ora found: {tnsnames_path}")
            # Print the geodepot entry from tnsnames.ora
            try:
                with open(tnsnames_path, 'r') as f:
                    content = f.read()
                    if 'geodepot' in content.lower():
                        print("Geodepot entry found in tnsnames.ora")
                    else:
                        print("WARNING: No geodepot entry found in tnsnames.ora")
            except Exception as e:
                print(f"Error reading tnsnames.ora: {e}")
        else:
            print(f"ERROR: tnsnames.ora not found at {tnsnames_path}")
    else:
        print(f"ERROR: TNS_ADMIN directory does not exist: {tns_admin}")
    
    # Check if required Oracle DLLs exist
    if oracle_home and os.path.isdir(oracle_home):
        print(f"ORACLE_HOME directory exists: {oracle_home}")
        
        # Check in both oracle_home and oracle_home/bin
        oci_dll_paths = [
            os.path.join(oracle_home, 'oci.dll'),
            os.path.join(oracle_home, 'bin', 'oci.dll')
        ]
        
        found_dll = False
        for path in oci_dll_paths:
            if os.path.isfile(path):
                print(f"oci.dll found: {path}")
                found_dll = True
                
        if not found_dll:
            print(f"ERROR: oci.dll not found in {oracle_home} or {os.path.join(oracle_home, 'bin')}")
    else:
        print(f"ERROR: ORACLE_HOME directory does not exist: {oracle_home}")
    
    print("=== End of Oracle Configuration ===\n")

if __name__ == "__main__":
    check_oracle_config()
    
    from multiprocessing import freeze_support
    freeze_support()
    main()
