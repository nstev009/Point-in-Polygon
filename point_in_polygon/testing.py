import os
import time

tns_admin = r"C:\or"
print(f"Checking access to {tns_admin}...")

if os.path.exists(tns_admin):
    print(f"✓ Path exists")
    print(f"Checking for tnsnames.ora...")
    tnsnames_path = os.path.join(tns_admin, 'tnsnames.ora')
    
    if os.path.exists(tnsnames_path):
        print(f"✓ tnsnames.ora found")
        try:
            with open(tnsnames_path, 'r') as f:
                content = f.read(4000)  # Read first 1000 chars
                print(f"Content preview: {content[:100]}...")
                if 'geodepot' in content.lower():
                    print("✓ 'geodepot' entry found in tnsnames.ora")
                else:
                    print("✗ No 'geodepot' entry found in tnsnames.ora")
        except Exception as e:
            print(f"✗ Error reading tnsnames.ora: {e}")
    else:
        print(f"✗ tnsnames.ora not found")
else:
    print(f"✗ Path does not exist or is not accessible")
    
print("Waiting 10 seconds before exit...")
time.sleep(10)