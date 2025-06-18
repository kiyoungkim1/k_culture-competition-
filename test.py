import re


text = "영어에서 'big mouth'라는 표현과 비슷한 의미를 가진 한국의 관용표현은 무엇인가요? \\n 1\\t 손이 크다  2\\t 발이 넓다  3\\t 입이 싸다  4\\t 입이 무겁다"

# choices
choices_list = []
items = re.findall(r'(\d+)\s*\\t\s*([^0-9]+)', text)
for num, phrase in items:
    choices_list.append(f"{num}) {phrase.strip()}")
choices = '\n'.join(choices_list)


print(choices)