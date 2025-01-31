import cx_Oracle
from sqlalchemy import create_engine, text, pool
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Tuple, Optional, Dict, Any
import keyring as kr
import os
import sys

# Handle imports differently when running as main
if __name__ == '__main__':
    # Add the parent directory to sys.path for direct script execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from point_in_polygon.setup_keyring import setup_keyring
else:
    from .setup_keyring import setup_keyring

import psutil
import multiprocessing

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


def list_tables(hostname, database_name):
    """List available tables in the database."""
    try:
        # Get credentials
        cred = kr.get_credential(hostname, "")
        if not cred:
            print(f"No credentials found for {hostname}. Setting up now...")
            setup_keyring(hostname)
            cred = kr.get_credential(hostname, "")
            if not cred:
                print("Failed to get credentials")
                return

        # Create connection string
        connection_str = f"oracle+cx_oracle://{cred.username}:{cred.password}@{hostname}"
        engine = create_engine(connection_str)

        # Query to list tables
        query = f"""
        SELECT table_name 
        FROM all_tables 
        WHERE owner = '{database_name}'
        AND table_name LIKE '%NGD%'
        ORDER BY table_name
        """

        print(f"\nListing tables in {database_name} schema containing 'NGD':")
        with engine.connect() as connection:
            result = connection.execute(text(query))
            for row in result:
                print(f"- {row[0]}")

    except Exception as e:
        print(f"Error listing tables: {str(e)}")


def check_database_connection(
    hostname: str = "Geodepot",
    username: Optional[str] = None,
    password: Optional[str] = None,
    port: int = 1521,
    database_name: str = "WAREHOUSE",
    keyring_service: str = "Geodepot"
) -> Tuple[bool, str]:
    """
    Check the database connection by attempting to connect and run a simple query.
    
    Args:
        hostname: Database hostname (default: Geodepot)
        username: Database username (optional if using keyring)
        password: Database password (optional if using keyring)
        port: Database port number
        database_name: Oracle database name (default: WAREHOUSE)
        keyring_service: Service name used to store credentials in keyring
    
    Returns:
        Tuple[bool, str]: (success status, message)
    """
    try:
        # If credentials not provided, try to get from keyring
        if not (username and password):
            # First try to get the username from environment or use default
            if not username:
                username = "wjeanph"  # default username if not specified
            password = kr.get_password(keyring_service, username)
            if not password:
                return False, "No credentials found in keyring. Please set up credentials first."

        # Create connection string - match the original implementation
        connection_str = f"oracle+cx_oracle://{username}:{password}@{hostname}"
        
        # Create engine with connection pooling
        engine = create_engine(connection_str, poolclass=pool.QueuePool)
        
        # Try to connect and execute a simple query
        with engine.connect() as connection:
            # Use a very lightweight query that just returns the current timestamp
            # DUAL is a special table in Oracle that exists in the default schema
            result = connection.execute(text("SELECT SYSTIMESTAMP FROM DUAL"))
            timestamp = result.scalar()
            
            return True, f"Successfully connected to database. Current timestamp: {timestamp}"
            
    except SQLAlchemyError as e:
        error_msg = str(e).split('\n')[0]  # Get first line of error
        return False, f"Database connection failed: {error_msg}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def check_system_specs() -> Dict[str, Any]:
    """
    Check system specifications relevant for multiprocessing optimization.
    
    Returns:
        Dict containing system specifications:
        - cpu_count: Number of logical CPU cores
        - cpu_count_physical: Number of physical CPU cores
        - memory_total_gb: Total system memory in GB
        - memory_available_gb: Available system memory in GB
        - disk_usage: Dictionary with disk usage information
        - cpu_percent: Current CPU utilization percentage
        - memory_percent: Current memory utilization percentage
    """
    try:
        # Get CPU information
        cpu_count = multiprocessing.cpu_count()
        cpu_count_physical = psutil.cpu_count(logical=False)
        
        # Get memory information
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024 ** 3)  # Convert to GB
        memory_available_gb = memory.available / (1024 ** 3)
        
        # Get disk information for the current working directory
        disk_usage = psutil.disk_usage(os.getcwd())
        disk_info = {
            'total_gb': disk_usage.total / (1024 ** 3),
            'used_gb': disk_usage.used / (1024 ** 3),
            'free_gb': disk_usage.free / (1024 ** 3),
            'percent_used': disk_usage.percent
        }
        
        # Get current CPU and memory utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = memory.percent
        
        specs = {
            'cpu_count': cpu_count,
            'cpu_count_physical': cpu_count_physical,
            'memory_total_gb': round(memory_total_gb, 2),
            'memory_available_gb': round(memory_available_gb, 2),
            'disk_usage': {k: round(v, 2) if isinstance(v, float) else v 
                          for k, v in disk_info.items()},
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent
        }
        
        # Print formatted output
        print("\n=== System Specifications ===")
        print(f"CPU Cores (Logical/Physical): {cpu_count}/{cpu_count_physical}")
        print(f"Memory Total: {specs['memory_total_gb']:.2f} GB")
        print(f"Memory Available: {specs['memory_available_gb']:.2f} GB")
        print(f"Current CPU Usage: {cpu_percent}%")
        print(f"Current Memory Usage: {memory_percent}%")
        print("\nDisk Information (Working Directory):")
        print(f"Total: {disk_info['total_gb']:.2f} GB")
        print(f"Used: {disk_info['used_gb']:.2f} GB ({disk_info['percent_used']}%)")
        print(f"Free: {disk_info['free_gb']:.2f} GB")
        print("===========================\n")
        
        return specs
        
    except Exception as e:
        print(f"Error checking system specifications: {str(e)}")
        return {}


if __name__ == "__main__":
    # Check system specifications
    specs = check_system_specs()
    
    # Example usage of database connection
    success, message = check_database_connection()
    if success:
        print("Database connection successful!")
    else:
        print(f"Database connection failed: {message}")
