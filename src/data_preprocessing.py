import pandas as pd
import json
import re

def get_domain_specific_info():
    with open('resource/data_given/korean_culture_qa_V1.0_train.json', 'r', encoding='utf-8') as f:
        data_train = json.load(f)

    with open('resource/data_given/korean_culture_qa_V1.0_dev.json', 'r', encoding='utf-8') as f:
        data_dev = json.load(f)

    with open('resource/data_given/korean_culture_qa_V1.0_test.json', 'r', encoding='utf-8') as f:
        data_test = json.load(f)

    def parse_korean_list(text):
        # 정규 표현식으로 번호와 내용을 추출
        items = re.findall(r'\d+\)\s*[^0-9]+', text)
        return [item.strip() for item in items]

    # category_list = [] # {'문화 지식', '문화 관점', '문화 실행'}
    # domain_list = [] # {'가치관', '역사', '사회', '지리', '과학기술', '정치/경제', '교육', '예술', '풍습/문화유산', '일상생활'}
    domain_specific = {'가치관':[], '역사':[], '사회':[], '지리':[], '과학기술':[], '정치/경제':[], '교육':[], '예술':[], '풍습/문화유산':[], '일상생활':[]}

    data_all = data_train+data_dev#+data_test
    for ele in data_all:
        category = ele['input']['category']
        domain = ele['input']['domain']
        question_type = ele['input']['question_type']
        topic_keyword = ele['input']['topic_keyword']
        question = '   '.join(parse_korean_list(ele['input']['question'].replace('\\t', ')').replace('\\n', '\n')))

        answer = ele['output']['answer'] if 'output' in ele else None
        if answer:
            if '#' in answer:
                answer = answer.split('#')[0]

        domain_specific[domain].append(f"""질문: {question}
답변: {answer}""")

    for key, value in domain_specific.items():
        domain_specific[key] = '\n\n###\n\n'.join(value)

    return domain_specific


# domain_specific = get_domain_specific_info()
# print(domain_specific)


