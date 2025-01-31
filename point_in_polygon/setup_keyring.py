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
    
    Args:
        hostname: The hostname of the database (default: "Geodepot")
        
    Returns:
        bool: True if credentials were stored successfully, False otherwise
        
    Example:
        >>> from point_in_polygon.setup_keyring import setup_keyring
        >>> setup_keyring()  # Will prompt for username and password
        >>> # Or specify a different hostname
        >>> setup_keyring(hostname="CustomHost")
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
        
        # Store both username and password in the keyring using service name
        print("Storing username...")
        kr.set_password(service_name, "username", username)
        print("Storing password...")
        kr.set_password(service_name, "password", password)

        print("Verifying stored credentials...")
        # Verify the credentials were stored correctly
        retrieved_username = kr.get_password(service_name, "username")
        print(f"Retrieved username: {retrieved_username}")
        retrieved_password = kr.get_password(service_name, "password")
        print(f"Password retrieved: {'Yes' if retrieved_password else 'No'}")
        
        if retrieved_username != username:
            print("Error: Username verification failed")
            return False
        if retrieved_password != password:
            print("Error: Password verification failed")
            return False

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
