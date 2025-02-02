import urllib.request
import tensorflow as tf
from gd import download_and_load_gpt2

url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"
)

# If this doesn't run and complains of distutils uninstall and
# reinstall setuptools pkg if Python version is >= 3.12
if __name__ == "__main__":
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    print(f"\nParams: {params.keys()}")
    print(f"Settings: {settings}")
    print(f"Token embedding layer weight tensor dimensions: {params["wte"].shape}")

    # tf.random.set_seed(0)
