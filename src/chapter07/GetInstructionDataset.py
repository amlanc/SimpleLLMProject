import json
import os
import urllib.request


file_path = "../../data/instruction-data.json"


def download_and_load_file(file_path):
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    
    if not os.path.exists(file_path):
        print("download_and_load_file(): Downloading file from " + url)
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        
        print("download_and_load_file(): Writing to " + file_path)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
            file.close()
    else:
        print("download_and_load_file(): File already exists at " + file_path)
        pass
        # with open(file_path, "r", encoding="utf-8") as file:
        #     text_data = file.read()
        #     print("download_and_load_file(): File content: \n", text_data)
        #     file.close()
    
    
    #   Testing implementation
    if os.path.exists(file_path):
        print("download_and_load_file(): Returning file content of  " + file_path)
        with open(file_path, "r") as file:
            data = json.load(file)
            # print(data)
            return data

def main():
    data = download_and_load_file(file_path)
    print("Number of entries:", len(data))
    print(f"Example entry: {data[50]}")
    
if __name__ == "__main__":
    main()