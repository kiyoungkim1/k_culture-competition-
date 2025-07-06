import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

from src.data import CustomDataset
from src.data import make_chat, make_validation
from src.post_processing import apply_post_processing


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, required=True, help="input filename")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
# fmt: on

def check_vram(device):
    # 현재 사용 중인 GPU 번호
    device = torch.device(device)

    # 현재 할당된 VRAM
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB 단위
    # 현재 캐시된 VRAM (PyTorch가 메모리 관리를 위해 유지)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB 단위

    print(f"Allocated VRAM: {allocated:.2f} MB")
    print(f"Reserved VRAM: {reserved:.2f} MB")

def main(args):
    # Prepare model loading kwargs
    model_kwargs = {
        "device_map": args.device,  # ex: "cuda" or "auto"
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,  # optional: bfloat16도 가능
    }

    if args.use_auth_token:
        model_kwargs["use_auth_token"] = args.use_auth_token

    # model = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        **model_kwargs,

    )
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    
    # Prepare tokenizer loading kwargs
    tokenizer_kwargs = {}
    if args.use_auth_token:
        tokenizer_kwargs["use_auth_token"] = args.use_auth_token
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        # tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]
    #
    # if len(tokenizer.encode("\n\n", add_special_tokens=False)) == 1:    # \n\n 자체가 하나의 token일 수도 있음
    #     terminators.append(tokenizer.convert_tokens_to_ids("\n\n"))
    #
    # class StopOnDoubleNewline(StoppingCriteria):
    #     def __init__(self, tokenizer, stop_sequence="\n\n"):
    #         self.stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
    #
    #     def __call__(self, input_ids, scores, **kwargs):
    #         if input_ids.shape[1] >= len(self.stop_ids):
    #             if input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids:
    #                 return True
    #         return False
    # stop_criteria = StoppingCriteriaList([
    #     StopOnDoubleNewline(tokenizer)
    # ])

    file_test = args.input
    # dataset = CustomDataset(file_test, tokenizer)
    #
    with open(file_test, "r", encoding='utf8') as f:
        result = json.load(f)

    # for idx in tqdm.tqdm(range(len(dataset))):
    for idx, example in tqdm.tqdm(enumerate(result)):
        # 1.1 답변 생성
        message_chat = make_chat(example["input"])

        source = tokenizer.apply_chat_template(
            message_chat,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )
        input_ids = source[0]

        outputs = model.generate(
            input_ids.to(args.device).unsqueeze(0),
            max_new_tokens=1536,
            eos_token_id=tokenizer.eos_token_id, #terminators,
            # stopping_criteria=stop_criteria,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=0.8,
            temperature=0.3,
            top_p=0.8,
            do_sample=True,
        )
        output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # 1.2 postprocessing
        output_processed = apply_post_processing(output_text)

        # 2.1 validation
        message_val = make_validation(example["input"], output_processed)

        source = tokenizer.apply_chat_template(
            message_val,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )
        input_ids = source[0]

        output_validation = model.generate(
            input_ids.to(args.device).unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id, #terminators,
            # stopping_criteria=stop_criteria,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=0.8,
            temperature=0.3,
            top_p=0.8,
            do_sample=True,
        )
        output_validataion = tokenizer.decode(output_validation[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # 2.2 post_processing
        output_final = apply_post_processing(output_validataion)


        result[idx]["output"] = {
            "raw": output_text,
            "answer_before_validation": output_processed,
            "validation": output_validataion,
            "answer": output_final,
        }

        # log
        print("output_text", output_text)
        print("output_processed", output_processed)
        print("validation", validation)
        print("output_processed", output_processed)
        check_vram(args.device)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))