# Arrhythmia-Detector

## 딥러닝 기반 부정맥 감지 시뮬레이션

<br>

- 👪 개발자
    - 한승우

- 📌 협업, 관리도구
    - Git, Agile

- 📱 APP
    - Language & Environment: Python 3.7.10, PyCharm(전체적인 개발), VScode(터미널 실행 테스트)

    - Frameworks : PyQt5, PyTorch

    - Developer : 한승우


## 1. 개요

- 2015년 기준, 유엔의 보고서에 따르면 세계는 고령화 시대로 접어들고 있음
- 또한 사람은 보통 나이가 들 수록 심혈관 시스템이 약해지며 부정맥과 같은 심혈관 질환에 취약해진다고 알려져 있음

- 본 프로젝트는 생체신호의 원시 신호만을 이용하고 딥러닝을 활용하여 부정맥 자동감지를 수행

![image](https://user-images.githubusercontent.com/38157496/118226098-44bf2f00-b4c1-11eb-8137-b26af09451d2.png)

## 2. 설명

### 2-1. 환경 설계

<br>

1. 전체 구조

> ![image](https://user-images.githubusercontent.com/38157496/101754340-fcb56e00-3b16-11eb-8a86-13edd155fc72.png)

> 학습된 모델을 이용하여 GUI 내부에서 전처리 및 자동감지 수행

<br>

2. 딥러닝 구조

> ![image](https://user-images.githubusercontent.com/38157496/118346119-4433a080-b574-11eb-878b-b095468f6407.png)

> PyTorch의 Conv1D API를 활용하여 딥러닝 설계

<br>

### 2-2. 핵심 기능

<br>

1. 화면

![image](https://user-images.githubusercontent.com/38157496/118346535-46e3c500-b577-11eb-92af-76721fc43ebc.png)

- (1) 파일 불러오기 기능

    - 경로의 위치를 파악하여 뒤에서 3번째 단어에 따라 (예) .csv -> c ) mat, csv, pkl 불러오기 가능

- (2) 부정맥 감지 기능

    - 사전 학습된 PyTorch 모델을 토대로 부정맥일 경우 부정맥이 감지

    - 고정 사이즈 기반 신호 분할 후 모델에 입력

- (3) 감지 이력 기능

    - 부정맥이 감지될 경우 현재의 시간이 기록

    - 신호가 끝까지 불러왔을 경우 로그를 저장

- (4) 운용 기능
    - 실행 , 정지, 초기화, 종료 기능
    - matplotlib의 애니메이션 기능 활용
    - python의 global 기능을 활용한 스위치 설정으로 실행/정지/초기화 가능

- (5) 디스플레이 기능
    - PyQt5의 Widget 및 matplotlib의 canvas API 활용

<br>

2. 모델 학습

- (1) trainer의 train 함수 구현

    - loss (손실)이 적을때 모델을 새로 저장한 뒤 다음과 같이 터미널에서 출력, 이후 log 폴더에 모델 학습 로그 저장

```python
Training start.
----------------------------------------------------------------------------------------------------
Epoch : 1
----------------------------------------------------------------------------------------------------
Save Model. Iteration : 0, Loss : 0.7419296503067017
Save Model. Iteration : 1, Loss : 0.7344326972961426
Save Model. Iteration : 2, Loss : 0.7135477662086487
... (중략)
Epoch : 5
----------------------------------------------------------------------------------------------------
Save Model. Iteration : 965, Loss : 5.184491601539776e-05
----------------------------------------------------------------------------------------------------
Training finish.
```

- (2) Slack 연동

    - knockknock API와 Slack의 Webhook API를 활용하여
    학습이 종료될 경우 메세지 출력

![image](https://user-images.githubusercontent.com/38157496/118346298-67ab1b00-b575-11eb-9bb8-45fa7cdebde3.png)

<br>

Q. TensorFlow가 아닌 PyTorch를 적용한 이유

TensorFlow의 Keras 프레임워크를 선택하였으나 모델 inference 시 잠시 로딩이 걸려 빠른 추론이 불가능하다고 생각해 PyTorch 적용

<br>

## 3. 실행 방법

- 아나콘다 가상환경 등으로 python 3.7 버전 환경 설정 뒤에 ```pip install -r requirements.txt``` 명령어로 라이브러리 설치

- python main.py 실행 후 출력되는 값
```
1 : 프로그램 실행       2 : 모델 학습   3 : 종료        (숫자 입력)
```
에 맞는 값 실행 시 프로그램 작동

슬랙 Webhook은 본인의 URL 주소 및 채널을 입력해야 함 (참고문헌 6 참고)

## 4. 참고문헌

[1] United Nations. Department of economic and social affairs population division. World population aging 2015. New York, 2015

[2] Y. Li, J. Bisera, M. Weil, and W. Tang, “An algorithm used for ventricular fibrillation detection without interrupting chest compression,” IEEE Trans. Biomed. Eng., vol. 59, no. 1, pp. 78–86, Jan. 2012.

[3] B. M. Asl, S. K. Setarehdan, and M. Mohebbi, “Support vector machine-based arrhythmia classification using reduced features of heart rate variability signal,” Artificial Intelligence in Medicine, vol. 44, no. 1, pp. 51–64, 2008.

[4] N. V. Thakor, Y. S. Zhu, and K. Y. Pan, “Ventricular tachycardia and fibrillation detection by a sequential hypothesis testing algorithm,” IEEE Trans. Biomed. Eng., vol. 37, no. 9, pp. 837–43, Sep. 1990

[5] S. Kiranyaz, T. Ince, and M. Gabbouj, “Real-time patient-specific ECG classification by 1-D convolutional neural networks,” IEEE Transactions
on Biomedical Engineering, vol. 63, no. 3, pp. 664–675, 2015.

[6] https://jojoldu.tistory.com/552
