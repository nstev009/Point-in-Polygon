import keyring as kr
import keyring.util.platform_


def get_service_name(hostname: str) -> str:
    """Get the service name for keyring storage."""
    return f"point_in_polygon_{hostname.lower()}"


def list_keyring_contents():
    try:
        # Get the current keyring backend being used
        backend = kr.get_keyring()
        print(f"Current keyring backend: {backend.__class__.__name__}")

        # Ask for the hostname to check
        hostname = input("Enter the hostname to check (default: Geodepot): ").strip() or "Geodepot"
        service_name = get_service_name(hostname)
        
        # Try to retrieve the credentials
        username = kr.get_password(service_name, "username")
        if username:
            password = kr.get_password(service_name, "password")
            print(f"\nFound credentials in service: {service_name}")
            print(f"Username: {username}")
            print(f"Password: {'[STORED]' if password else '[NOT FOUND]'}")
        else:
            print(f"\nNo credentials found in service: {service_name}")

    except Exception as e:
        print(f"Error accessing keyring: {str(e)}")


def clear_keyring_contents():
    try:
        hostname = input("Enter the hostname to clear credentials (default: Geodepot): ").strip() or "Geodepot"
        service_name = get_service_name(hostname)
        
        # Get current username to delete its password
        username = kr.get_password(service_name, "username")
        
        # Delete credentials
        if username:
            try:
                kr.delete_password(service_name, "username")
                kr.delete_password(service_name, "password")
                print(f"Successfully cleared credentials for service: {service_name}")
            except Exception as e:
                print(f"Error clearing credentials: {str(e)}")
        else:
            print(f"No credentials found for service: {service_name}")
            
    except Exception as e:
        print(f"Error accessing keyring: {str(e)}")


if __name__ == "__main__":
    while True:
        print("\nKeyring Test Menu:")
        print("1. List credentials")
        print("2. Clear credentials")
        print("3. Exit")
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            list_keyring_contents()
        elif choice == "2":
            clear_keyring_contents()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")
