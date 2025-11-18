# pip install -U transformers huggingface_hub torch accelerate sentencepiece safetensors

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
huggingface_token = os.environ.get("HF_TOKEN")
def load_or_download_model(model_id, local_dir):
    """Load model from local if exists, otherwise download from Hugging Face."""
    if not os.path.exists(local_dir):
        print(f"Model not found locally. Downloading '{model_id}'...")
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=huggingface_token
        )
        print("✅ Download complete.")
    else:
        print("✅ Model already exists locally.")
    
    # Load the model and tokenizer after download or if already present
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(local_dir)
    return model, tokenizer


if __name__ == "__main__":

    # MODEL_ID = "google/gemma-3-4b-it"
    # LOCAL_DIR = "./models/gemma-3-4b-it"

    # MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    # LOCAL_DIR = "./models/Meta-Llama-3-8B-Instruct"

    # MODEL_ID = "google/gemma-3-270m-it"
    # LOCAL_DIR = "./google/gemma-3-270m-it"

    MODEL_ID = "bigscience/bloomz-560m"
    LOCAL_DIR = "./bigscience/bloomz-560m"

    # MODEL_ID = "Qwen/Qwen3-0.6B"
    # LOCAL_DIR = "./Qwen/Qwen3-0.6B"

    try:
        model, tokenizer = load_or_download_model(model_id=MODEL_ID, local_dir=LOCAL_DIR)
    except Exception as e:
        print(f"An error occurred: {e}")