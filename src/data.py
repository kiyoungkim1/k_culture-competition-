import json
import re

import torch
from torch.utils.data import Dataset

from src.data_preprocessing import get_domain_specific_info

# domain_specific_info = get_domain_specific_info()

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다.
무의미하게 동일한 단어나 문장을 반복하여 작성하지 말아주세요."""

        with open(fname, encoding='utf-8') as f:
            data = json.load(f)

        def make_chat(inp):
            # domain
            domain = inp['domain']
            # domain_info = domain_specific_info[domain]
            ## TODO: domain info가 문제의 키워드 들이랑은 다른 지식인데 들어가는게 꼭 좋은지 모르겠음. 패턴을 가르쳐 주는게 아니기 때문에 많이 알려주는게 의미가 없을 듯.

            if inp['question_type'] == '선다형':
                # question
                question = inp['question'].split('\\n')[0].strip()

                # choices
                choices_list = []
                items = re.findall(r'(\d+)\s*\\t\s*([^0-9]+)', inp['question'])
                for num, phrase in items:
                    choices_list.append(f"{num}) {phrase.strip()}")
                choices = ' '.join(choices_list)

                chat = """질문을 잘 읽고 한국사람으로써 가장 적절한 답변을 찾아주세요. 답변은 숫자만 작성해야합니다.
예시는 해당 분야의 한국 문화에 대한 문제를 정확하게 푼 것이다.
이와 같이 한국 문화를 기반으로 주어진 문제를 풀어야 한다.

<예시>
질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?
1) 주사위 놀이   2) 검무   3) 격구   4) 영고   5) 무애무
답변: 3
</예시>

질문: {}
{}

topic_keyword는 {}입니다.
먼저 <background> </background> 사이에 topic_keyword와 관련하여 중요한 정보 3가지를 총 150자 이내, 질문 선택지의 1, 2, 3, 4, 5번의 중요한 정보 3가지를 각각 150자 이내로 작성해 주세요.
<background>정보를 바탕으로  <solve> </solve> 사이에 답변에 해당하는 숫자를 작성하고 그 이유에 대해 작성하세요..
이 후 <eval> </eval>에서 <solve>에서 작성한 답이 맞는지 평가하고, 맞다면 왜 맞는지, 틀리다면 왜 틀렸는지에 대해 간단히 작성하세요.
상기 내용을 바탕으로 최종 답변인 숫자는 <result> </result> 사이에 작성해 주세요. 맞으면 <solve>에서 작성한 숫자를, 아니면 <eval>의 피드백을 바탕으로 정답을 새로 작성해 주세요. 숫자만 작성해야 합니다.""".format(question, choices, inp['topic_keyword'])

            elif inp['question_type'] == '서술형':
                chat = """[질문]을 잘 읽고 한국사람으로써 가장 적절한 답변을 작성해 주세요. 질문에 대한 답변을 완성된 문장으로 서술하시오.
예시는 해당 분야의 한국 문화에 대한 문제를 정확하게 푼 것이다.
이와 같이 한국 문화를 기반으로 주어진 문제를 풀어야 한다.

<예시>
질문: 대한민국의 행정구역 체계를 서술하세요.
답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다.
</예시>

질문: {}

topic_keyword는 {}입니다.
먼저 <background> </background> 사이에 topic_keyword와 관련하여 중요한 정보 3가지를 150자 이내로 작성하고, 주어진 질문의 인물, 사건, 장소등을 모두 골라 해당 내용에 관련한 내용을 3가지씩 각각 150자로 작성해 주세요.
<background>내용을 바탕으로 <solve> </solve> 사이에 답변에 해당하는 답을 완성된 문장으로 작성하세요.
이 후 <eval> </eval>에서 그 답이 맞는지 평가하고, 맞다면 왜 맞는지, 틀리다면 왜 틀렸는지에 대해 간단히 작성하세요.
상기 내용을 참고하여 최종적으로 완성된 문장의 답변을 <result> </result> 사이에 작성해 주세요. 맞으면 <solve>에서 작성한 숫자를, 아니면 <eval>의 피드백을 바탕으로 정답을 새로 작성해 주세요.""".format(inp['question'], inp['topic_keyword'])

            elif inp['question_type'] == '단답형':
                chat = """[질문]을 잘 읽고 한국사람으로써 가장 적절한 답변을 작성해 주세요. 질문에 대한 답을 2단어 이내로 간단히 답하시오.
예시는 해당 분야의 한국 문화에 대한 문제를 정확하게 푼 것이다.
이와 같이 한국 문화를 기반으로 주어진 문제를 풀어야 한다.

<예시>
질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?
답변: 정약용
</예시>

질문: {}

topic_keyword는 {}입니다.
먼저 <background> </background> 사이에 topic_keyword와 관련하여 중요한 정보 3가지를 150자 이내로 작성하고, 주어진 질문의 인물, 사건, 장소등을 모두 골라 해당 내용에 관련한 내용을 3가지씩 각각 150자로 작성해 주세요.
<background>내용을 바탕으로 <solve> </solve> 사이에 답변에 해당하는 답을 2단어 이내로 작성하고 그 이유에 대해 작성해 주세요..
이 후 <eval> </eval>에서 그 답이 맞는지 평가하고, 맞다면 왜 맞는지, 틀리다면 왜 틀렸는지에 대해 간단히 작성하세요.
상기 내용을 참고하여 최종적으로 2단어 이내의 답변을 <result> </result> 사이에 작성해 주세요. 맞으면 <solve>에서 작성한 숫자를, 아니면 <eval>의 피드백을 바탕으로 정답을 새로 작성해 주세요.""".format(inp['question'], inp['topic_keyword'])

            return chat
        
        for example in data:
            user_prompt = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if False: print(f'[DBG] message: {message}')
     
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


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
