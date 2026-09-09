import os

os.environ.setdefault("HF_HOME", os.path.join(os.environ["WORK"], "hf_cache"))

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Prior-Labs/tabpfn_3", filename="tabpfn-v3-classifier-v3_default.ckpt")
print(model_path)
