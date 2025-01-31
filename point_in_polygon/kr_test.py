import keyring
import keyring.util.platform_


def list_keyring_contents():
    try:
        # Get the current keyring backend being used
        backend = keyring.get_keyring()
        print(f"Current keyring backend: {backend.__class__.__name__}")

        # Ask for the hostname and username to check
        hostname = input("Enter the hostname to check: ").strip()
        username = input("Enter the username to check: ").strip()

        # Try to retrieve the password
        password = keyring.get_password(hostname, username)
        if password:
            print(f"\nFound credentials for {username}@{hostname}")
            print(f"Password: {password}")
        else:
            print(f"\nNo credentials found for {username}@{hostname}")

    except Exception as e:
        print(f"Error accessing keyring: {str(e)}")


if __name__ == "__main__":
    list_keyring_contents()
