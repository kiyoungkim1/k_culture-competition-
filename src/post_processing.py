import json
import re
import pandas as pd

def apply_post_processing(answer_text):
    if "<result>" in answer_text and "</result>" in answer_text:
        result_match = re.search(r"<result>\s*(.*?)\s*</result>", answer_text, re.DOTALL)

        if result_match:
            extracted_result = result_match.group(1).strip()
        else:
            print("ERROR")
            extracted_result = "# ERROR"

    elif "<result>" in answer_text and "</result>" not in answer_text: # <result>는 있는데 </result>는 없는 경우가 있음
        extracted_result = answer_text.split('<result>')[-1].strip()

    elif "<result>" not in answer_text and "</result>" in answer_text:
        extracted_result = answer_text.split('</result>')[-1].strip()

    else:
        extracted_result = answer_text.split('\n')[-1].strip()

    # final cleaning
    extracted_result = extracted_result.replace('**','').replace('\n', ' ')
    if ':' in extracted_result:  # '"answer": ~~~' 이런식으로 답이 나오는 경우가 있음
        extracted_result = extracted_result.split(':')[-1]

    extracted_result = extracted_result.strip()

    return extracted_result


def get_result_excel(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    data_list = []
    for ele in json_data:
        id = ele['id']
        answer_before_validation = ele['output']['answer_before_validation'] if 'answer_before_validation' in ele['output'] else None
        answer = ele['output']['answer']

        data_list.append({
            'id': id,
            'before_valid({})'.format(json_path): answer_before_validation,
            'answer({})'.format(json_path): answer
        })

    if 'korean_culture_qa_V1.0' in json_path:
        pd.DataFrame(data_list).to_excel('json_result.xlsx', index=False)
    else:
        pd.DataFrame(data_list).to_excel('json_result_{}.xlsx'.format(json_path.split('.')[0]), index=False)


if __name__ == '__main__':
    get_result_excel('resource/data_given/korean_culture_qa_V1.0_dev.json')
    get_result_excel('result_exaone_32B.json')
    get_result_excel('result_ax70B.json')

# python src/post_processing.py
