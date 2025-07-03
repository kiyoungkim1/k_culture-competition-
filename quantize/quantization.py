from transformers import AutoTokenizer
from hqq.engine.hf import HQQModelForCausalLM

# 모델 ID
model_id = "skt/A.X-4.0"

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# HQQ로 2-bit 양자화
model = HQQModelForCausalLM.from_pretrained(
    model_id,
    quantization_bit=2,
    group_size=16,      # 그룹 최소화로 정확도 개선
    axis=0,             # axis=0이 정확도에 유리하지만 inference는 axis=1에 최적화
    device_map="auto",
)

# 바로 inference 가능
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
