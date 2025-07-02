import re

# question = "새마을운동의 41주년을 맞아 새마을의 날이 국가 기념일로 제정된 날짜는 언제인가요? \\n 1\\t3월 15일 2\\t4월 22일 3\\t5월 1일 4\\t6월 6일 5\\t10월 9일"
question = "한국에서 봉건적 굴레와 일제 침략으로부터의 해방을 제시하며 1927년에 창립된 항일 여성운동 단체의 이름은? \\n 1\\t신간회 2\\t어영청 3\\t근우회 4\\t훈련도감 5\\t대한독립여성단"

question = question.replace("\\t", ") ")
choice_text = question.split('\\n')[-1]
print(choice_text)

choices = re.findall(r'\d+\)\s*.*?(?=\s+\d+\)|$)', choice_text)
choices = [c.strip() for c in choices]
choices = '\n'.join(choices)
print(choices)