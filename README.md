# Point in Polygon

A Python package for efficient point-in-polygon operations using GeoPandas and Oracle Spatial database. This package is designed to replace ArcGIS for point-in-polygon operations, offering better performance and more flexibility.

## Features

- Fast point-in-polygon operations using GeoPandas
- Oracle Spatial database integration
- Chunked processing for large datasets
- Parallel processing support
- Caching mechanism for database queries
- Configurable CSV input/output
- Secure credential management using keyring

## Installation

```bash
pip install point_in_polygon
```

## Quick Start

### 1. Setting Up Database Credentials

Before using the package, you need to set up your database credentials using keyring. This is a one-time setup that securely stores your credentials:

```python
from point_in_polygon import setup_keyring

# Set up credentials for default hostname (Geodepot)
setup_keyring()

# Or specify a different hostname
setup_keyring(hostname="CustomHost")
```

The setup will:
1. Prompt for your username
2. Securely prompt for your password (input will be hidden)
3. Store credentials in your system's secure keyring
4. Verify the credentials were stored correctly

Example terminal interaction:
```
Setting up credentials for Geodepot
Enter your username: your_username
Enter your password: ********
Success: Credentials stored securely for your_username@Geodepot
```

### 2. Basic Usage

Here's a simple example of how to use the point-in-polygon functionality:

```python
from point_in_polygon import PointInPolygonConfig, point_in_polygon

# Create configuration
config = PointInPolygonConfig(
    # Required parameters
    csv_long_lat_file="input_points.csv",  # CSV with your points data
    output_csv_file="output_results.csv",   # Where to save the results
    table_name="YOUR_TABLE_NAME",           # Oracle table name
)

# Run the point-in-polygon operation
point_in_polygon(config)
```

### 3. Advanced Configuration

Here's a more detailed example showing all configuration options:

```python
from point_in_polygon import PointInPolygonConfig, point_in_polygon

config = PointInPolygonConfig(
    # Required input/output files
    csv_long_lat_file="input_points.csv",
    output_csv_file="output_results.csv",
    table_name="YOUR_TABLE_NAME",
    
    # Database connection settings (defaults shown)
    hostname="Geodepot",          # Default database hostname
    database_name="WAREHOUSE",    # Default database name
    
    # Database column and spatial settings
    uid="BB_UID",                # UID column name
    shape_column="SHAPE",        # Shape column name
    spatial_reference=3347,      # Spatial reference system code
    conditions=None,             # Optional SQL WHERE conditions
    
    # CSV column mapping
    id_column="rec_id",         # Name of the ID column in your CSV
    lat_column="latitude",      # Name of the latitude column in your CSV
    lon_column="longitude",     # Name of the longitude column in your CSV
    
    # Processing options
    chunk_size=100000,          # Process this many rows at once
    use_parallel=True,          # Enable parallel processing
    use_threads=False,          # Use processes instead of threads
    max_workers=None,           # Number of workers (None = auto)
    return_all_points=True,     # Include unmatched points in output
    match_status_column="bb_uid_matched",  # Column name for match status
    
    # Caching options
    use_bb_uid_cache=True,      # Cache database queries
    cache_dir="cache",          # Where to store cache files
    cache_max_age_days=120,     # How long to keep cache files
    
    # Testing options
    sample=False               # Whether to sample the data
)

point_in_polygon(config)
```

## Input CSV Format

Your input CSV file should have at least these columns (names can be configured):
- `rec_id`: Unique identifier for each point
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate

Example:
```csv
rec_id,latitude,longitude
1,45.4215,-75.6972
2,45.4216,-75.6973
```

## Output Format

The output CSV will contain:
- All columns from your input CSV
- A match status column (default name: "bb_uid_matched")
  - `True`: Point falls within a polygon
  - `False`: Point does not fall within any polygon

## Performance Tips

1. **Chunking**: Adjust `chunk_size` based on your available memory. Larger chunks process faster but use more memory.

2. **Parallel Processing**: 
   - Enable with `use_parallel=True`
   - Use `use_threads=False` for CPU-bound tasks
   - Set `max_workers` to control resource usage
   - Each worker process automatically gets its own copy of the geometric data
   - Database queries are parallelized with each worker having its own connection
   - Memory usage scales with the number of workers as each gets a data copy

3. **Caching**:
   - Enable with `use_bb_uid_cache=True`
   - Cache files are stored in `cache_dir`
   - Set `cache_max_age_days` to control cache freshness
   - Caching geometric data reduces database load for repeated operations

4. **Memory Management**:
   - The package loads geometric data once and distributes copies to worker processes
   - Memory usage = (size of geometric data) × (number of workers)
   - Monitor memory usage and adjust `max_workers` if needed
   - Use smaller `chunk_size` if processing very large datasets

## Troubleshooting

1. **Database Connection Issues**:
   - Ensure credentials are set up using `setup_keyring()`
   - Verify hostname and database name
   - Check network connectivity

2. **CSV Issues**:
   - Verify column names match your configuration
   - Ensure CSV has headers
   - Check for valid latitude/longitude values

3. **Performance Issues**:
   - Try adjusting `chunk_size`
   - Enable parallel processing
   - Use caching for repeated queries

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
