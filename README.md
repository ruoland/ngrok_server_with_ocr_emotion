감정분석과 OCR 분석을 처리하는 서버

실행하려면 torch, transformers, Flask, kss==5.2.0 설치 필요, 전부 다 CPU 기준
pip install torch transformers scikit-learn Flask kss==5.2.0 opencv-python paddlepaddle paddleocr flask_cors mmengine mmdet

git clone -b dev-3.x https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
pip install mmengine
pip install mmdet

pip install -U openmim
mim install mmcv

안되면
pip install mmcv-full

안되면
pip install mmcv

안되면
pip install mmcv==2.1.0

설치 되면 콘솔창에 cd.. 입력하기

그리고 python model_server.py 실행하여 모델의 서버 열고, VS Code에 새 터미널 생성한 후 ngrok http 5000(포트) 입력하기
그러면 ngrok 서버 주소가 나오는데 그 주소가 현재 이 프로그램 실행중인 컴퓨터 주소임

캘린더나 다이어리에 서버 주소가 잘 설정 되어 있다면 서버 실행만 하면 끝.
