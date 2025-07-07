import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

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
    with open(file_test, "r", encoding='utf8') as f:
        result = json.load(f)

    for idx, example in tqdm.tqdm(enumerate(result)):
        get_answer = False
        question_type = example['input']['question_type']

        # keyword가 정답인 경우
        if question_type == '선다형':
            if example['input']['topic_keyword'] in example['input']['question']:
                output_text = answer_in_it
                output_processed = answer_in_it
                output_validation = answer_in_it
                output_final = answer_in_it

                get_answer = True

        elif question_type == '단답형':
            message = [
                {
                    "role": "system",
                    "content": """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 한국 문화 박사입니다.

기존 학술 자료 또는 백과사전 기준으로 질문에 답해주세요.
확인된 사실 위주로 작성해 주세요. 가설이나 개인 의견은 포함하지 마세요.
출처 기반 정보만 활용해야 하며, 추론은 생략해 주세요."""
                },
                {"role": "user", "content": """[질문]을 잘 읽고 한국사람으로써 가장 적절한 답변을 작성한 것입니다.
이와 같이 한국 문화를 기반으로 주어진 문제를 풀어야 한다. 현대의 한반도 중 남한의 문화를 바탕으로 해야합니다.

질문: {}
답변: {}

해당 질문에 대한 답변이 맞나요? 너무 일반적인 답이 아니고, 50%이상 맞다면 <result> </result> tag안에 해당 답변을 그대로 작성해 주세요.
질문에 대한 답변으로 매우 적절하지 않다면 <result> </result> tag 안에 '적절하지 않음'이라고 작성해 주세요.
답변은 반드시 한국어로 작성해야 합니다. 반복하여 동일한 단어나 문장은 생성하지 말고 tag 내 결과 이외 다른 글은 작성하지 마세요.""".format(example['input']['question'], example['input']['topic_keyword'])},
            ]

            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )
            input_ids = source[0]

            outputs = model.generate(
                input_ids.to(args.device).unsqueeze(0),
                max_new_tokens=512,
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

            if example['input']['topic_keyword'] in output_processed:
                output_text = example['input']['topic_keyword']
                output_processed = example['input']['topic_keyword']
                output_validation = example['input']['topic_keyword']
                output_final = example['input']['topic_keyword']
                get_answer = True

        # keyword가 답이 아닌 경우
        if get_answer == False:
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

            outputs = model.generate(
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
            output_validation = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

            # 2.2 post_processing
            output_final = apply_post_processing(output_validation)


        result[idx]["output"] = {
            "raw": output_text,
            "answer_before_validation": output_processed,
            "validation": output_validation,
            "answer": output_final,
        }

        # log
        print("output_text", output_text)
        print("output_processed", output_processed)
        print("output_validation", output_validation)
        print("output_final", output_final)
        check_vram(args.device)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))