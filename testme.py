from point_in_polygon.main import PointInPolygonConfig, point_in_polygon
import sys
from multiprocessing import freeze_support


def main():
    # Input and output file paths
    input_file = r"\\fld6filer\Record_Linkage\Point_in_Polygon_Examples\input_test_data\bg_latlongs_100_recs_with_header.csv"
    output_file = "output_results.csv"


    # List available tables first
    # list_tables("Geodepot", "WAREHOUSE")

    # Create configuration
    config = PointInPolygonConfig(
        csv_long_lat_file=input_file,
        output_csv_file=output_file,
        table_name="WC2026NGD_A_202412",  # Updated to latest available version
        database_name="WAREHOUSE",  # Schema is specified here
        id_column="bg_sn",
        lat_column="bg_latitude",
        lon_column="bg_longitude",
        chunk_size=50,  # Smaller chunk size for small dataset
        use_parallel=True,
        return_all_points=True,
        use_bb_uid_cache=True,  # Enable caching
        cache_dir="cache",  # Cache directory
        cache_max_age_days=120  # Cache expiry in days
    )


    try:
        # Run the point-in-polygon analysis
        point_in_polygon(config)
        print(f"Analysis complete. Results saved to {output_file}")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    freeze_support()
    main()