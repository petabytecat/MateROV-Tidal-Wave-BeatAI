import json
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def download_image(url, image_path):
    if not os.path.exists(image_path):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(image_path, "wb") as img_file:
                    img_file.write(response.content)
                print(f"Image downloaded successfully to {image_path}")
                return True
            else:
                print(f"Failed to retrieve the image. Status code: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"Error downloading to {image_path}: {e}")
            return False
    else:
        print(f"Image already exists at {image_path}. Skipping download.")
        return False

def process_json_file(json_path):
    # Extract the number from the filename
    filename = os.path.basename(json_path)
    number = ''.join(filter(str.isdigit, filename.split('.')[0]))

    if not number:
        print(f"Could not extract number from filename: {filename}")
        return

    # Create directory for this number
    output_dir = os.path.join("downloaded_images", number)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(json_path, "r") as file:
            data = json.load(file)

        # Create tasks for ThreadPoolExecutor
        download_tasks = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            for file_data in data:
                if isinstance(file_data, dict) and 'uuid' in file_data and 'url' in file_data:
                    image_path = os.path.join(output_dir, f"{file_data['uuid']}.png")
                    download_tasks.append(
                        executor.submit(download_image, file_data['url'], image_path)
                    )

            # Wait for all downloads to complete
            for future in as_completed(download_tasks):
                future.result()

    except json.JSONDecodeError as e:
        print(f"Error reading JSON file {json_path}: {e}")
    except Exception as e:
        print(f"Error processing file {json_path}: {e}")

def main():
    # Create base directory for all downloads
    os.makedirs("downloaded_images", exist_ok=True)

    # Get all JSON files in the data directory
    data_dir = Path("data")
    json_files = list(data_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files to process")

    # Process each JSON file
    for json_file in json_files:
        print(f"\nProcessing {json_file}")
        process_json_file(json_file)

if __name__ == "__main__":
    main()
