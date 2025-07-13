import argparse
import json
import tqdm
import re
import random

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

def get_llm_result(model, tokenizer, args, message_chat, max_new_tokens=1024, temperature=0.5, top_p=0.8, repetition_penalty=0.8):
    source = tokenizer.apply_chat_template(
        message_chat,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    )
    input_ids = source[0]

    outputs = model.generate(
        input_ids.to(args.device).unsqueeze(0),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,  # terminators,
        # stopping_criteria=stop_criteria,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        no_repeat_ngram_size=5,
    )
    output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # postprocessing
    output_processed = apply_post_processing(output_text)

    return output_processed, output_text

def main(args):
    print('model_id: {}'.format(args.model_id))

    # Prepare model loading kwargs
    model_kwargs = {}

    if args.use_auth_token:
        model_kwargs["use_auth_token"] = args.use_auth_token

    if 'autoround' in args.model_id:
        model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                     device_map=args.device, torch_dtype="auto")

    elif 'GPTQ' in args.model_id or 'fp8' in args.model_id:   # 양자화 없음
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            device_map=args.device
        )

    elif 'Midm' in args.model_id:  # 16bit 양자화
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            device_map=args.device,

            torch_dtype=torch.float16
        )

    elif 'aaaaa' in args.model_id:  # 8bit 양자화
        model_kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            device_map=args.device,

            load_in_8bit=True
        )

    else:  # 4 bit 양자화
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            device_map=args.device,

            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
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
        # TODO: 다른 파라미터에서 3번 생성하고 마지막에 3개 비교해서 답 내기.
        get_answer = False
        question_type = example['input']['question_type']

        import difflib
        def find_most_similar_index(choices_list, topic_keyword):
            # 띄어쓰기 제거
            topic = topic_keyword.replace(' ', '')
            choices_no_space = [choice.replace(' ', '') for choice in choices_list]
            # difflib로 유사도 계산 후 가장 높은 index 반환
            similarities = [difflib.SequenceMatcher(None, topic, choice).ratio() for choice in choices_no_space]
            return similarities.index(max(similarities))

        # keyword가 정답인 경우
        if question_type == '선다형':
            choice_text = example['input']['question'].replace("\\t", ")").split('\\n')[-1]
            choices = re.findall(r'\d+\)\s*.*?(?=\s+\d+\)|$)', choice_text)
            choices_list = [c.strip() for c in choices]
            choices = '   '.join(choices_list)

            if example['input']['topic_keyword'] in choices:
                answer_idx = find_most_similar_index(choices_list, example['input']['topic_keyword'])+1

                output_text = answer_idx
                output_processed = answer_idx

            else:
                message_chat = make_chat(example["input"])
                for i in range(5):
                    output_processed, output_text = get_llm_result(model, tokenizer, args, message_chat,
                                                                   max_new_tokens=1024, temperature=round(random.uniform(0.3, 0.5), 2), top_p=round(random.uniform(0.6, 0.8), 2))
                    try:
                        int(output_processed)
                        break
                    except:
                        pass

        elif question_type == '단답형':
            message = [
                {
                    "role": "system",
                    "content": """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 한국 문화 박사입니다.
주어진 질문에 대한 답변이 정확한지 판단해 주세요."""
                },
                {"role": "user", "content": """질문을 잘 읽고 한국사람으로써 가장 적절한 답변을 작성한 것입니다.

질문: {}
답변: {}

해당 질문에 대한 답변이 구체적이고 정확하다면 <result> </result> tag안에 해당 답변을 그대로 작성해 주세요.
질문에 대한 답변이 너무 포괄적이라면(답이 '한강'인데 '강'이라고 한다거나) <result> </result> tag 안에 '포괄적임'이라고 작성해 주세요.
이 이외의 다른 텍스트는 작성하지 마세요.""".format(example['input']['question'], example['input']['topic_keyword'])},
            ]

            output_processed, output_text = get_llm_result(model, tokenizer, args, message, max_new_tokens=256,
                                                           temperature=round(random.uniform(0.5, 0.7), 2), top_p=round(random.uniform(0.7, 0.8), 2), repetition_penalty=1.5)

            if example['input']['topic_keyword'] in output_processed:
                output_text = example['input']['topic_keyword']
                output_processed = example['input']['topic_keyword']

            else:
                # 1.1 답변 생성
                for i in range(5):
                    message_chat = make_chat(example["input"])
                    output_processed, output_text = get_llm_result(model, tokenizer, args, message_chat,
                                                                   max_new_tokens=1024, temperature=round(random.uniform(0.5, 0.7), 2), top_p=round(random.uniform(0.7, 0.8), 2))
                    if len(output_processed) < 15:
                        break

        elif question_type=="서술형":
            message_chat = make_chat(example["input"])
            output_processed, output_text = get_llm_result(model, tokenizer, args, message_chat,
                                                           max_new_tokens=1536, temperature=round(random.uniform(0.5, 0.7), 2), top_p=round(random.uniform(0.7, 0.8), 2))

        result[idx]["output"] = {
            "raw": output_text,
            "answer": output_processed
        }

        # log
        print("output_processed", output_processed)
        check_vram(args.device)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))