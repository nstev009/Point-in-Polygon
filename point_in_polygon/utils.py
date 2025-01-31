import cx_Oracle
from sqlalchemy import create_engine, text, pool
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Tuple, Optional
import keyring as kr
from .setup_keyring import setup_keyring

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

if __name__ == "__main__":
    # Example usage
    success, message = check_database_connection()
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
