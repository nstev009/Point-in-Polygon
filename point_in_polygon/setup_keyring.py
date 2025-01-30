"""
Setup keyring for storing database credentials securely.
"""

import getpass
import keyring as kr


def setup_keyring(hostname: str = "Geodepot") -> bool:
    """
    Set up keyring with database credentials.
    
    Args:
        hostname: The hostname of the database (default: "Geodepot")
        
    Returns:
        bool: True if credentials were stored successfully, False otherwise
        
    Example:
        >>> from point_in_polygon import setup_keyring
        >>> setup_keyring()  # Will prompt for username and password
        >>> # Or specify a different hostname
        >>> setup_keyring(hostname="CustomHost")
    """
    try:
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

        # Store the password in the keyring
        kr.set_password(hostname, username, password)

        # Verify the credentials were stored correctly
        retrieved_password = kr.get_password(hostname, username)
        if retrieved_password != password:
            print("Error: Failed to store credentials correctly")
            return False

        print(f"Success: Credentials stored securely for {username}@{hostname}")
        return True

    except Exception as e:
        print(f"Error setting up credentials: {str(e)}")
        return False


if __name__ == "__main__":
    setup_keyring()
