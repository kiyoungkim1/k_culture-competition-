import json
import re
import pandas as pd

def apply_post_processing(answer_text):
    if "<result>" in answer_text and "</result>" in answer_text:
        result_match = re.search(r"<result>\s*(.*?)\s*</result>", answer_text, re.DOTALL)

        if result_match:
            extracted_result = result_match.group(1).strip()
        else:
            print("ERROR", item['id'])
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
        answer_before_validation = ele['output']['answer_before_validation']
        answer = ele['output']['answer']

        data_list.append({
            'id': id,
            'before_valid({})'.format(json_path): answer_before_validation,
            'answer({})'.format(json_path): answer
        })

    pd.DataFrame(data_list).to_excel('json_result_{}.xlsx'.format(json_path.split('.')[0]), index=False)

if __name__ == '__main__':
    # get_result_excel('result.json')
    get_result_excel('result_deepseekR1_32B.json')


# python src/post_processing.py
