import cx_Oracle
import os
import keyring as kr
import time
import shutil
from pathlib import Path

def get_service_name(hostname):
    return f"point_in_polygon_{hostname.lower()}"

def setup_oracle_connection():
    """Set up everything needed for Oracle TCPS connection"""
    print("\n=== Oracle Connection Setup ===\n")
    
    # Set base Oracle environment variables
    os.environ['ORACLE_HOME'] = r"C:\ora19c\product\19.0.0\client_2"
    os.environ['PATH'] = os.environ.get('PATH', '') + r";C:\ora19c\product\19.0.0\client_2\bin"

    # 4. Check if wallet files exist or prompt user for them
    wallet_files = ["cwallet.sso", "ewallet.p12"]
    missing_files = [f for f in wallet_files if not os.path.exists(os.path.join(wallet_dir, f))]
    
    if missing_files:
        print(f"\n⚠️ Missing required wallet files: {', '.join(missing_files)}")
        print("For TCPS connections, you need Oracle wallet files.")
        print("Please ask your database administrator for these files and place them in:")
        print(f"  {wallet_dir}")
        print("\nAttempting alternate connection methods...\n")
    else:
        print(f"✓ Required wallet files found in {wallet_dir}")
    
    # 5. Set TNS_ADMIN to our wallet directory
    os.environ['TNS_ADMIN'] = wallet_dir
    print(f"\n✓ Set TNS_ADMIN={wallet_dir}")
    
    return wallet_dir

def test_connection_methods():
    """Try multiple connection methods to Oracle"""
    hostname = "geodepot"
    service_name = get_service_name(hostname)
    
    # Get credentials
    username = kr.get_password(service_name, "username")
    password = kr.get_password(service_name, "password")
    
    if not username or not password:
        print("No credentials found. Please check keyring.")
        return False
    
    # Try all connection methods
    connection_methods = [
        # Method 1: EZ Connect
        {
            "name": "EZ Connect (TCP)",
            "connect_fn": lambda: cx_Oracle.connect(
                f"{username}/{password}@exa-db-prod1-3sduj-scan.scocip1client.scocivcn.oraclevcn.com:1521/geodepot_s.scocip1client.scocivcn.oraclevcn.com"
            )
        },
        # Method 2: Standard TNS
        {
            "name": "TNS Entry",
            "connect_fn": lambda: cx_Oracle.connect(
                user=username,
                password=password,
                dsn="geodepot",
                encoding="UTF-8"
            )
        },
        # Method 3: Full connection string with TCPS
        {
            "name": "Full TCPS Connection String",
            "connect_fn": lambda: cx_Oracle.connect(
                f"{username}/{password}@(DESCRIPTION=(ADDRESS=(PROTOCOL=TCPS)(HOST=exa-db-prod1-3sduj-scan.scocip1client.scocivcn.oraclevcn.com)(PORT=2485))(CONNECT_DATA=(SERVICE_NAME=geodepot_s.scocip1client.scocivcn.oraclevcn.com)))"
            )
        },
        # Method 4: TCP instead of TCPS (try port 1521)
        {
            "name": "TCP Connection (port 1521)",
            "connect_fn": lambda: cx_Oracle.connect(
                f"{username}/{password}@(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=exa-db-prod1-3sduj-scan.scocip1client.scocivcn.oraclevcn.com)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=geodepot_s.scocip1client.scocivcn.oraclevcn.com)))"
            )
        }
    ]
    
    # Try each method
    success = False
    for method in connection_methods:
        try:
            print(f"\nTrying {method['name']}...")
            connection = method["connect_fn"]()
            print(f"✅ Connection successful with {method['name']}!")
            print(f"Database version: {connection.version}")
            
            # Test a simple query
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()
            print(f"Test query result: {result}")
            
            cursor.close()
            connection.close()
            success = True
            
            # Store the successful method for future use
            print(f"\n✅ SUCCESS: Use {method['name']} for your connection")
            
            # Break after first successful method
            break
            
        except Exception as e:
            print(f"❌ {method['name']} failed: {str(e)}")
    
    return success

def update_main_py(successful_method):
    """Update main.py with the successful connection method"""
    # TODO: Implement if a method succeeds
    pass

if __name__ == "__main__":
    print("Testing Oracle connection with multiple methods")
    wallet_dir = setup_oracle_connection()
    success = test_connection_methods()
    
    if not success:
        print("\n❌ All connection methods failed!")
        print("\nRecommended actions:")
        print("1. Confirm the database is accessible (ask your DBA)")
        print("2. Confirm the host can be reached (ping the hostname)")
        print("3. Verify Oracle instant client installation and configuration")
        print("4. Check for required wallet files for TCPS connections")
        print("5. Try connecting with Oracle SQL Developer to validate credentials")
    
    input("\nPress Enter to exit...")