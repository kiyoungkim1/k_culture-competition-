# pip install transformers hqq bitsandbytes accelerate huggingface_hub

# 2bit 양자화
from transformers import AutoTokenizer, AutoModelForCausalLM
import hqq
import torch

model_id = "UNIVA-Bllossom/DeepSeek-llama3.3-Bllossom-70B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# HQQ quantization (e.g., 4bit)
hqq.quantize_model(model, bits=2, group_size=64)

# 로컬에 저장
save_dir = "./llama3.3-hqq-quant"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


# # hf repo에 저장
# #huggingface-cli login  # 토큰 필요 (https://huggingface.co/settings/tokens)
#
# from huggingface_hub import create_repo, upload_folder
#
# repo_name = "your-username/llama3.3-hqq-quant"
# create_repo(repo_name, private=True)  # 또는 public=True
#
# upload_folder(
#     folder_path=save_dir,
#     repo_id=repo_name,
#     commit_message="Upload HQQ-quantized LLaMA3.3 model"
# )
#
# # 다시 불러오기
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model = AutoModelForCausalLM.from_pretrained(
#     "your-username/llama3.3-hqq-quant",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained("your-username/llama3.3-hqq-quant")
