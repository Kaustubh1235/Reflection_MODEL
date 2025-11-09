from huggingface_hub import snapshot_download
import shutil
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--google--flan-t5-large")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Flan-T5-Large model deleted from cache.")
else:
    print("Model not found in cache.")
