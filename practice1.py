# 해당 코드에는 6가지의 오류가 있습니다.
# 인당 1개씩 오류를 찾은 후, add_practice branch의 compare&pull request로 들어가 comment 달아주세요.
# 오류 꼭 인당 1개씩만 찾아주세요^^
# 다른 사람이 comment 달지 않은 오류에만 comment 달아주세요.
# 최종적으로 모든 오류 수정 후 fix_error_자기이름 branch 생성 후 해당 branch에 push하고 pr(compare&pull request에 글 남기기)하기

import pandas as pd

d_a = [['권유진', '18학번', 24], ['김정하', '19학번', 23], ['나요셉', '18학번', 25], ['이경욱', '18학번', 24], ['윤경서', '19학번', 23], ['이예진', '19학번', 23], ['이수빈', '20학번', 24]]
df = pd.DataFrame(d_a)
df.columns = ['이름', '학번', '나이'] # 이수빈 변경
df['역할'] = ['학회장','부학회장','부하','부하','부하','부하','부하'] # 나요셉 변경

# 입학년도 계산하는 함수
def ent_year(x):
    return int('20' + x[:2]) # 해당 line에 오류 있음. 문자열로 합친 후 int로
df['입학년도'] = df['학번'].map(ent_year) # 해당 line에 오류 있음. 적용할 칼럼 변경

# 학년
df['학년'] = df['입학년도'].map(lambda x: 2022 - x - 1) # 해당 line에 오류 있음. lambda 계산 바르게 변경
df['졸업예정'] = df['학년'].map(lambda x: 2022 + (4-x) + 1) # 해당 line에 오류 있음. lambda 계산 바르게 변경

print(df)
