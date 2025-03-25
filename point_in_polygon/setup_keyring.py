"""
Setup keyring for storing database credentials securely.
"""

import getpass
import keyring as kr
import sys


def get_service_name(hostname: str) -> str:
    """Get the service name for keyring storage."""
    return f"point_in_polygon_{hostname.lower()}"


def setup_keyring(hostname: str = "Geodepot") -> bool:
    """
    Set up keyring with database credentials.
    """
    try:
        service_name = get_service_name(hostname)
        
        # Get username and password securely
        print(f"\nSetting up credentials for {hostname}")
        username = input("Enter your username: ").strip()
        if not username:
            print("Error: Username cannot be empty")
            return False
            
        # Use getpass for secure password input (won't show on screen)
        password = getpass.getpass("Enter your password: ").strip()
        if not password:
            print("Error: Password cannot be empty")
            return False

        print("Storing credentials...")
        
        # Method 1: Store as separate username/password entries (for get_password)
        print("Storing username and password separately...")
        kr.set_password(service_name, "username", username)
        kr.set_password(service_name, "password", password)

        # Method 2: Store in the format expected by get_credential()
        print("Storing credentials for direct retrieval...")
        kr.set_password(service_name, username, password)

        print("Verifying stored credentials...")
        # Verify the credentials were stored correctly (both methods)
        retrieved_username = kr.get_password(service_name, "username")
        retrieved_password = kr.get_password(service_name, "password")
        
        # Also verify the get_credential method will work
        credential = kr.get_credential(service_name, None)
        
        print(f"Retrieved username: {retrieved_username}")
        print(f"Password retrieved: {'Yes' if retrieved_password else 'No'}")
        print(f"Credential retrieval: {'Success' if credential and credential.username == username else 'Failed'}")
        
        if retrieved_username != username or retrieved_password != password:
            print("Error: Username/password verification failed")
            return False
        
        if not credential or credential.username != username or credential.password != password:
            print("Warning: Credential object verification failed")
            # Don't return False here, as some backends might not support get_credential

        print(f"Success: Credentials stored securely for {username}@{hostname}")
        return True

    except Exception as e:
        print(f"Error setting up credentials: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    if not setup_keyring():
        sys.exit(1)


if __name__ == "__main__":
    main()
