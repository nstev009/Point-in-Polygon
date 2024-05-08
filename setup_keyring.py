import keyring as kr


def main():
    # Get inputs from the user
    hostname = input("Enter the hostname: ").strip()
    username = input("Enter the username: ").strip()
    password = input("Enter the password: ").strip()

    # Check if any of the fields are empty
    if not hostname or not username or not password:
        print("All fields (hostname, username, and password) must be filled out.")
        return

    # Store the password in the keyring
    kr.set_password(hostname, username, password)

    # Retrieve the credential from the keyring to confirm it was saved
    retrieved_password = kr.get_password(hostname, username)
    if retrieved_password is None:
        print("Failed to store credentials.")
    else:
        print(f"Credentials stored successfully for {username}@{hostname}")


if __name__ == "__main__":
    main()
