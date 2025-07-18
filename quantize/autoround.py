from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
# import torch

model_name = "skt/A.X-4.0"
print(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
# model = torch.compile(model) # torch compile
tokenizer = AutoTokenizer.from_pretrained(model_name)

bits, group_size, sym = 2, 128, True

# # normal
# autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

## the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

## 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
## format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format='auto_round')