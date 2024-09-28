감정분석과 OCR 분석을 처리하는 서버

실행하려면 torch, transformers, Flask, kss.core 설치 필요
pip install torch, transformers, Flask, kss.core
ngrok_server.py 실행하여 서버 열고, VS Code에 새 터미널 생성한 후 ngrok http 5000(포트) 입력하기
그러면 ngrok 서버 주소가 나오는데 그 주소가 현재 이 프로그램 실행중인 컴퓨터 주소임
그 주소를 엣지펑션 깃허브(https://github.com/ruoland/emotion_edge_function) 의 설명 따라 바꾸고 

readme 에 있는 Edge Function 설정(https://github.com/Liana10042024/j-day) 한 후 엣지펑션 배포 후 다이어리 앱에서 실행하기

배포 명령어! 그 뒤 슈퍼베이스 접속 후 개인 프로젝트 들어간 후 엣지펑션 배포 됐는지 확인하기
npx supabase functions deploy huggingface-model

