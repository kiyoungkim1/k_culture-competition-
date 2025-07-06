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

    # category_list = [] # {'문화 지식', '문화 관점', '문화 실행'}
    # domain_list = [] # {'가치관', '역사', '사회', '지리', '과학기술', '정치/경제', '교육', '예술', '풍습/문화유산', '일상생활'}
    domain_specific = {'가치관':[], '역사':[], '사회':[], '지리':[], '과학기술':[], '정치/경제':[], '교육':[], '예술':[], '풍습/문화유산':[], '일상생활':[]}

    data_all = data_train+data_dev#+data_test
    for ele in data_all:
        category = ele['input']['category']
        domain = ele['input']['domain']
        question_type = ele['input']['question_type']
        topic_keyword = ele['input']['topic_keyword']

        question = ele['input']['question'].split('\\n')[0].strip()

        if '\\t' not in ele['input']['question']:
            question_considering_choice = question
        else:
            # 선다형
            choice_text = ele['input']['question'].replace("\\t", ")").split('\\n')[-1]
            choices = re.findall(r'\d+\)\s*.*?(?=\s+\d+\)|$)', choice_text)
            choices_list = [c.strip() for c in choices]
            choices = '   '.join(choices_list)

            question_considering_choice = f"""{question}
{choices}"""

        answer = ele['output']['answer'] if 'output' in ele else None

        # answer
        if answer:
            if '#' in answer:
                answer = answer.split('#')[0]

        domain_specific[domain].append(f"""질문: {question_considering_choice}
답변: {answer}""")

    for key, value in domain_specific.items():
        domain_specific[key] = '\n\n###\n\n'.join(value)

    return domain_specific


# domain_specific = get_domain_specific_info()
# print(domain_specific)


