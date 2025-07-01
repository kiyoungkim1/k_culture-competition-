import json
import re

# JSON 데이터 예시 (변수명 data에 리스트 형태로 저장)
with open('result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# <result> 내부의 텍스트를 추출하고 output 필드를 갱신
for item in data:
    answer_text = item.get("output", {}).get("answer", "")

    if "</result>" in answer_text:
        result_match = re.search(r"<result>\s*(.*?)\s*</result>", answer_text, re.DOTALL)

        if result_match:
            extracted_result = result_match.group(1).strip()
        else:
            print("ERROR")
            extracted_result = "# ERROR"

    elif "<result>" in answer_text: # <result>는 있는데 </result>는 없는 경우가 있음
        extracted_result = answer_text.split('<result>')[-1].strip()

    else:
        extracted_result = answer_text.split('\n')[-1].strip()

    # final cleaning
    extracted_result = extracted_result.replace('**','').replace('\n', ' ')
    if ':' in extracted_result:  # '"answer": ~~~' 이런식으로 답이 나오는 경우가 있음
        extracted_result = extracted_result.split(':')[-1]

    item["output"] = {"answer": extracted_result}

# 결과 출력 또는 저장
with open('output_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# python src/post_processing.py
