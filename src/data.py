import json
import re

import torch
from torch.utils.data import Dataset

from src.data_example import get_domain_specific_info

domain_specific_info = get_domain_specific_info()

def make_chat(inp):
    # domain
    domain = inp['domain']
    domain_info = domain_specific_info[domain]
    ## TODO: domain info가 문제의 키워드 들이랑은 다른 지식인데 들어가는게 꼭 좋은지 모르겠음. 패턴을 가르쳐 주는게 아니기 때문에 많이 알려주는게 의미가 없을 듯.

    if inp['question_type'] == '선다형':
        # question
        question = inp['question'].split('\\n')[0].strip()

        # choices
        choice_text = inp['question'].replace("\\t", ")").split('\\n')[-1]
        choices = re.findall(r'\d+\)\s*.*?(?=\s+\d+\)|$)', choice_text)
        choices_list = [c.strip() for c in choices]
        choices = '   '.join(choices_list)

        chat = """질문을 잘 읽고 한국사람으로써 가장 적절한 답변을 찾아주세요. 답변은 숫자만 작성해야합니다.
예시는 해당 분야의 한국 문화에 대한 문제를 정확하게 푼 것이다.
이와 같이 한국 문화를 기반으로 주어진 문제를 풀어야 한다.
주어진 선택지 중 가장 질문에 가까운 답을 찾아야 한다.

<예시>
질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?
1) 주사위 놀이   2) 검무   3) 격구   4) 영고
답변: 3

###

{}
</예시>

질문: {}
{}

topic_keyword는 {}입니다.
먼저 <background> </background> 사이에 질문과 관련하여 중요한 정보 3가지를 총 300자 이내, 질문과 관련하여 선택지의 1, 2, 3, 4번의 중요한 정보를 작성하고 각 선택지가 한국 사람의 입장에서 질문의 정답에 가장 적합한지에 대해 각각 150자로 작성해 주세요.
먼저 <background> </background> 사이에 질문과 관련하여 중요한 정보 3가지를 총 300자 이내, 질문과 관련하여 선택지의 1, 2, 3, 4번의 중요한 정보를 작성하고 각 선택지가 한국 사람의 입장에서 질문의 정답에 가장 적합한지에 대해 각각 150자로 작성해 주세요.
<background>정보를 바탕으로 <solve> </solve> 사이에 답변에 해당하는 숫자를 작성하고 그 이유에 대해 작성하세요.
이 후 <eval> </eval>에서 <solve>에서 작성한 답이 맞는지 평가하고, 맞다면 왜 맞는지, 틀리다면 왜 틀렸는지에 대해 간단히 작성하세요.
상기 내용을 바탕으로 최종 답변인 숫자는 <result> </result> 사이에 작성해 주세요. 맞으면 <solve>에서 작성한 숫자를, 아니면 <eval>의 피드백을 바탕으로 정답을 새로 작성해 주세요. 숫자만 작성해야 합니다.
일반적인 지식이 아니라, 반드시 전통적인 한국의 문화를 기반으로 풀어야 합니다.""".format(domain_info, question, choices, inp['topic_keyword'])

    elif inp['question_type'] == '서술형':
        chat = """[질문]을 잘 읽고 한국사람으로써 가장 적절한 답변을 작성해 주세요. 질문에 대한 답변을 완성된 문장으로 서술하시오.
예시는 해당 분야의 한국 문화에 대한 문제를 정확하게 푼 것으로 서술형 답변은 350~400글자 사이로 작성해야 합니다.
이와 같이 한국 문화를 기반으로 주어진 문제를 풀어야 한다.

<예시>
질문: 대한민국의 행정구역 체계를 서술하세요.
답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다.

###

{}
</예시>

질문: {}

topic_keyword는 {}입니다.
먼저 <background> </background> 사이에 질문에 관한 내용을 3가지를 각각 150자씩 총 500자 정도로 작성해 주세요.
<background>내용을 바탕으로 <solve> </solve> 사이에 답변에 해당하는 답을 완성된 문장으로 작성하세요.
이 후 <eval> </eval>에서 그 답이 맞는지 평가하고, 맞다면 왜 맞는지, 틀리다면 왜 틀렸는지에 대해 간단히 작성하세요.
상기 내용을 참고하여 최종적으로 완성된 문장의 답변을 <result> </result> 사이에 작성해 주세요. 맞으면 <solve>에서 작성한 글을, 아니면 <eval>의 피드백을 바탕으로 정답을 새로 작성해 주세요.
답변은 반드시 한국어로 작성해야 합니다. 350~400자 사이의 줄글로 작성해야 합니다.
일반적인 지식이 아니라, 반드시 전통적인 한국의 문화를 기반으로 풀어야 합니다.""".format(domain_info, inp['question'], inp['topic_keyword'])

    elif inp['question_type'] == '단답형':
        chat = """[질문]을 잘 읽고 한국사람으로써 가장 적절한 답변을 작성해 주세요. 질문에 대한 답을 2단어 이내로 간단히 답하시오.
예시는 해당 분야의 한국 문화에 대한 문제를 정확하게 푼 것이다.
이와 같이 한국 문화를 기반으로 주어진 문제를 풀어야 한다.

<예시>
질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?
답변: 정약용

###

{}
</예시>

질문: {}

topic_keyword는 {}입니다.
먼저 <background> </background> 사이에 주어진 질문에 대해 관련한 내용 3가지를 각각 150자씩 총 500자 정도로 작성해 주세요.
<background>내용을 바탕으로 <solve> </solve> 사이에 답변에 해당하는 답을 2단어 이내로 작성하고 그 이유에 대해 작성해 주세요..
이 후 <eval> </eval>에서 그 답이 맞는지 평가하고, 맞다면 왜 맞는지, 틀리다면 왜 틀렸는지에 대해 간단히 작성하세요.
상기 내용을 참고하여 최종적으로 2단어 이내의 답변을 <result> </result> 사이에 작성해 주세요. 맞으면 <solve>에서 작성한 답변을, 아니면 <eval>의 피드백을 바탕으로 정답을 새로 작성해 주세요. 2단어 이내로 작성해야합니다.
일반적인 지식이 아니라, 반드시 전통적인 한국의 문화를 기반으로 풀어야 합니다.""".format(domain_info, inp['question'], inp['topic_keyword'])

    message = [
        {
            "role": "system",
            "content": """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 한국 문화 박사입니다.

기존 학술 자료 또는 백과사전 기준으로 질문에 답해주세요.
확인된 사실 위주로 작성해 주세요. 가설이나 개인 의견은 포함하지 마세요.
출처 기반 정보만 활용해야 하며, 추론은 생략해 주세요.
무의미하게 동일한 단어나 문장을 반복하여 작성하지 말아주세요."""
        },
        {"role": "user", "content": chat},
    ]

    return message


def make_validation(inp, output):
    if inp['question_type'] == '선다형':
        # question
        question = inp['question'].split('\\n')[0].strip()

        # choices
        choice_text = inp['question'].replace("\\t", ")").split('\\n')[-1]
        choices = re.findall(r'\d+\)\s*.*?(?=\s+\d+\)|$)', choice_text)
        choices_list = [c.strip() for c in choices]
        choices = '   '.join(choices_list)

        chat = """한국 문화 박사로써 질문에 대한 선다형 문제 답변의 번호를 작성한 것이다.
주어진 선택지 중 가장 질문에 가까운 답을 찾은 것이다.

질문: {}
{}
답변: {}

먼저 <thinking> </thinking> tag 사이에 해당 질문에 대한 답변이 정확하다면 넘어가고, 정확하지 않다면 출처를 가져와 왜 정확하지 않고 실제 답변은 무엇인지 번호로 작성해 주세요.
상기 내용을 바탕으로 최종 답변인 숫자는 <result> </result> 사이에 작성해 주세요. 맞으면 원래 답변에서 작성한 숫자를, 아니면 <thinking>의 결과를 바탕으로 정답을 새로 작성해 주세요. 숫자만 작성해야 합니다.
일반적인 지식이 아니라, 반드시 전통적인 한국의 문화를 기반으로 풀어야 합니다.""".format(question, choices, output)

    elif inp['question_type'] == '서술형':
        chat = """한국 문화 박사로써 질문에 대한 선다형 문제 답변의 번호를 작성한 것이다.
주어진 선택지 중 가장 질문에 가까운 답을 찾은 것이다.

질문: {}
답변: {}

먼저 <thinking> </thinking> tag 사이에 해당 질문에 대한 답변이 정확하다면 넘어가고, 정확하지 않다면 출처를 가져와 왜 정확하지 않고 실제 답변은 무엇인지 새로운 답변으로 작성해 주세요.
상기 내용을 참고하여 최종적으로 완성된 문장의 답변을 <result> </result> 사이에 작성해 주세요. 맞으면 원래의 답변을을, 아니면 <thinking>의 결과를 바탕으로 정답을 새로 작성해 주세요. 답변은 반드시 한국어로 작성해야 합니다. 350~400자 사이의 줄글로 작성해야 합니다.
일반적인 지식이 아니라, 반드시 전통적인 한국의 문화를 기반으로 풀어야 합니다.""".format(inp['question'], output)

    elif inp['question_type'] == '단답형':
        chat = """한국 문화 박사로써 질문에 대한 선다형 문제 답변의 번호를 작성한 것이다.
주어진 선택지 중 가장 질문에 가까운 답을 찾은 것이다.

질문: {}
답변: {}

먼저 <thinking> </thinking> tag 사이에 해당 질문에 대한 답변이 정확하다면 넘어가고, 정확하지 않다면 출처를 가져와 왜 정확하지 않고 실제 답변은 무엇인지 2단어 이내의 답변으로 작성해 주세요.
상기 내용을 참고하여 최종적으로 2단어 이내의 답변을 <result> </result> 사이에 작성해 주세요. 맞으면 원래의 답변을, 아니면 <thinking>의 결과를 바탕으로 정답을 새로 작성해 주세요. 2단어 이내로 작성해야합니다.
일반적인 지식이 아니라, 반드시 전통적인 한국의 문화를 기반으로 풀어야 합니다.""".format(inp['question'], output)


    message = [
        {
            "role": "system",
            "content": """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 한국 문화 박사입니다.

    기존 학술 자료 또는 백과사전 기준으로 질문에 답해주세요.
    확인된 사실 위주로 작성해 주세요. 가설이나 개인 의견은 포함하지 마세요.
    출처 기반 정보만 활용해야 하며, 추론은 생략해 주세요.
    무의미하게 동일한 단어나 문장을 반복하여 작성하지 말아주세요."""
        },
        {"role": "user", "content": chat},
    ]

    return message


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 한국 문화 박사입니다.

기존 학술 자료 또는 백과사전 기준으로 정리해 주세요.
확인된 사실 위주로 작성해 주세요. 가설이나 개인 의견은 포함하지 마세요.
출처 기반 정보만 활용해야 하며, 추론은 생략해 주세요.
무의미하게 동일한 단어나 문장을 반복하여 작성하지 말아주세요."""

        with open(fname, encoding='utf-8') as f:
            data = json.load(f)

        for example in data:
            user_prompt = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_prompt},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )

            target = example.get("output", "")
            if target != "":
                target += tokenizer.eos_token
            target = tokenizer(
                target,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )

            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]

