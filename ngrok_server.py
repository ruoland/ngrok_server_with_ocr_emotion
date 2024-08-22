from flask import Flask, request
from emotion.emotion_run import emotion_run
from ocr.ocr_run import ocr_run
import json
import logging

#ngrok.exe http sought-shrew-miserably.ngrok-free.app 5000
#고정 도메인이 있다면 위 명령어 주소 변경해서 사용하기
#없다면 ngrok.exe http 포트
#하고 나온 주소를 넣기
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

@app.route('/emotion', methods=['POST'])
def emotion():
    app.logger.info("OCR request received")
    app.logger.debug(f"Request headers: {request.headers}")
    app.logger.debug(f"Request JSON: {request.json}")
    return emotion_run()
  
@app.route('/ocr', methods=['POST'])
def ocr():
    app.logger.info("OCR request received")
    app.logger.debug(f"Request headers: {request.headers}")
    app.logger.debug(f"Request JSON: {request.json}")
    response = ocr_run()
    app.logger.info(f"OCR response: {response}")
    return response

if __name__ == '__main__':
    app.run(debug=True)