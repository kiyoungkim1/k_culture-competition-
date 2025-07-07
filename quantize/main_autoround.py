from auto_round import AutoRoundForCausalLM, AutoRoundTokenizer

# 저장된 양자화 모델 경로
output_dir = "./tmp_autoround"

# 로딩
model = AutoRoundForCausalLM.from_quantized(output_dir)
tokenizer = AutoRoundTokenizer.from_pretrained(output_dir)

# 텍스트 생성
inputs = tokenizer("안녕", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
