"""
churn_prediction/ - segment_rules.json을 읽어 1·2·3단계 예측모델을
학습(train.py)하거나 저장된 모델로 추론(predict.py)하는 패키지.

이 패키지는 segment_discovery의 내부 구현을 알 필요가 없다 - 오직
segment_rules.json(인터페이스)만 의존한다.

train.py   : 재학습 (segment_discovery/app.py와 같은 묶음으로 가끔 실행)
predict.py : 추론 전용 (학습 없음, 자유 주기로 언제든 실행 가능)
"""
