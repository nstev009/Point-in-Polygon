import keyring
import keyring.backend
from point_in_polygon.setup_keyring import get_service_name

def check_keyring_setup():
    # Get the active keyring backend
    backend = keyring.get_keyring()
    print(f"Active keyring backend: {backend.__class__.__name__}")
    
    # Check where the backend is storing data
    if hasattr(backend, "file_path"):
        print(f"Keyring file location: {backend.file_path}")
    
    # List available services (if supported by the backend)
    try:
        hostname = "Geodepot"  # or whatever hostname you used
        service = get_service_name(hostname)
        username = keyring.get_password(service, "username")
        has_password = keyring.get_password(service, "password") is not None
        
        print(f"\nFound credentials:")
        print(f"Service: {service}")
        print(f"Username: {username}")
        print(f"Password stored: {'Yes' if has_password else 'No'}")
    except Exception as e:
        print(f"Error checking credentials: {str(e)}")

def check_using_get_credential():
    """Test using keyring.get_credential() API"""
    try:
        hostname = "Geodepot"  # or whatever hostname you used
        service = get_service_name(hostname)
        
        # get_credential() returns a credentials object with username and password
        credentials = keyring.get_credential(service, None)
        
        print("\nUsing keyring.get_credential():")
        if credentials:
            print(f"Service: {service}")
            print(f"Username: {credentials.username}")
            print(f"Password retrieved: {'Yes' if credentials.password else 'No'}")
        else:
            print(f"No credentials found for service: {service}")
    except Exception as e:
        print(f"Error using get_credential: {str(e)}")

if __name__ == "__main__":
    check_keyring_setup()
    check_using_get_credential()