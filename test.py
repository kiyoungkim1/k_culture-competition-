import re

def parse_korean_list(text):
    # 정규 표현식으로 번호와 내용을 추출
    items = re.findall(r'\d+\)\s*[^0-9]+', text)
    return [item.strip() for item in items]

# 사용 예시
input_text = """1) 손이 크다     2) 발이 넓다    3) 입이 싸다  4) 입이 무겁다"""
result = parse_korean_list(input_text)
print(result)
