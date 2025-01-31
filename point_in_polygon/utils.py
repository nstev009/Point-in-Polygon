import cx_Oracle
from sqlalchemy import create_engine, text, pool
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Tuple, Optional

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
            import keyring
            # First try to get the username from environment or use default
            if not username:
                username = "wjeanph"  # default username if not specified
            password = keyring.get_password(keyring_service, username)
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
