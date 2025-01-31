from point_in_polygon.main import PointInPolygonConfig, point_in_polygon
from point_in_polygon.setup_keyring import setup_keyring
import os
import keyring as kr
import sys

def check_oracle_client():
    """Check if Oracle Client is properly installed."""
    try:
        import cx_Oracle
        cx_Oracle.init_oracle_client()
        return True
    except Exception as e:
        print("\nError: Oracle Client library not found or not properly configured.")
        print("\nTo fix this, please follow these steps:")
        print("1. Download the Oracle Instant Client from:")
        print("   https://www.oracle.com/database/technologies/instant-client/winx64-64-downloads.html")
        print("2. Download the 'Basic Package' (e.g., instantclient-basic-windows.x64-21.12.0.0.0dbru.zip)")
        print("3. Extract the ZIP file to a permanent location (e.g., C:\\oracle\\instantclient_21_12)")
        print("4. Add the instant client directory to your PATH environment variable")
        print("   Or set ORACLE_HOME environment variable to the instant client directory")
        print("\nError details:", str(e))
        return False

# Input and output file paths
input_file = r"\\fld6filer\Record_Linkage\Point_in_Polygon_Examples\input_test_data\bg_latlongs_100_recs_with_header.csv"
output_file = "output_results.csv"

# Check Oracle Client first
if not check_oracle_client():
    sys.exit(1)

# Create configuration
config = PointInPolygonConfig(
    csv_long_lat_file=input_file,
    output_csv_file=output_file,
    table_name="WAREHOUSE.CENSUS_BLOCKS_2021",  # Adjust if needed
    id_column="bg_sn",
    lat_column="bg_latitude",
    lon_column="bg_longitude",
    chunk_size=1000,
    use_parallel=True,
    return_all_points=True
)

# Check if credentials exist in keyring
cred = kr.get_credential(config.hostname, "")
if not cred:
    print(f"No credentials found for {config.hostname}. Setting up now...")
    setup_keyring(config.hostname)
    cred = kr.get_credential(config.hostname, "")
    if not cred:
        print("Failed to set up credentials. Please try again.")
        sys.exit(1)

try:
    # Run the point-in-polygon analysis
    point_in_polygon(config)
    print(f"Analysis complete. Results saved to {output_file}")
except Exception as e:
    print(f"\nError during execution: {str(e)}")
    if "ORA-" in str(e):
        print("\nThis appears to be an Oracle database error. Please check:")
        print("1. Your database credentials")
        print("2. Database connectivity")
        print("3. That you have access to the specified table")
    sys.exit(1)