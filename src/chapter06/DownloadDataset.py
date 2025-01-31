#  Copyright (c) 2025 Amlan Chatterjee. All rights reserved.
#
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "../../data/"+"sms_spam_collection.zip"
extracted_path = "../../data/sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url,
                                 zip_path,
                                 extracted_path,
                                 data_file_path):
    print("Downloading and unzipping spam dataset...")
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    
    print("Didnt find existing spam dataset, ...")
    print("URL: ", url)
    print("ZIP path: ", zip_path)
    
    # Download the zip file
    get_zip_file(url, zip_path)
    
    # If Zip file is there and not extracted then extracted
    extract_zip_file(data_file_path, extracted_path, zip_path)


def extract_zip_file(data_file_path, extracted_path, zip_path):
    if not os.path.exists(extracted_path):
        print(f"{extracted_path} doesn't exist. Extracting...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)
                zip_ref.close()
                print("Extracted SPAM dataset successfully...")
                original_file_path = Path(extracted_path) / "SMSSpamCollection"
                print(f"Renaming {original_file_path} to {data_file_path}")
                os.rename(original_file_path, data_file_path)
                print(f"File downloaded and saved as {data_file_path}")
        except Exception as ex:
            print(f"Error extracting SPAM dataset \n {ex}")
    else:
        print(f"{extracted_path} file already exists. Skipping Extraction...")


def get_zip_file(url, zip_path):
    if not os.path.exists(zip_path):
        print(f"{zip_path} doesnt exist. Downloading...")
        try:
            with urllib.request.urlopen(url) as response:
                with open(zip_path, "wb") as out_file:
                    out_file.write(response.read())
                    out_file.flush()
                    out_file.close()
                    print(f"File downloaded at: {zip_path}")
        except Exception as e:
            print(f"Error downloading SPAM dataset \n {e}")
    else:
        print(f"{zipfile} file already exists. Skipping Download...")


def main():
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    

# Download SPAM Dataset
if __name__ == "__main__":
    main()
