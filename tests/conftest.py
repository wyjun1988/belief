"""테스트는 **합성 데이터만** 쓴다.

ADT 는 연구 라이선스 데이터라 CI 에 둘 수 없고, 무엇보다 실측 수치 대조는 회귀
검출기로 약하다 — 숫자가 변해도 "데이터가 달라졌나" 로 흐지부지된다. 대신 각
알고리즘의 **설계 의도를 입력으로 표현**해서, 의도가 깨지면 반드시 실패하게 한다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
