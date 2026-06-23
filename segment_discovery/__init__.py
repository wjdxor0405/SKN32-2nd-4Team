"""
segment_discovery/ - 분석A(세그먼트 경계)·분석B(위험속성)·서브트랙Q(risk_count)
를 실행해 segment_rules.json을 산출하는 패키지.

진입점: app.py (분석A → 분석B → 서브트랙Q 순서로 실행)
실행 주기: churn_prediction의 train.py와 같은 묶음(재학습 묶음) - 표본이
충분히 쌓였을 때 함께 실행.
"""
