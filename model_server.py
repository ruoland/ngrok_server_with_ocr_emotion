from flask import Flask, request, jsonify
from emotion.emotion_run import emotion_run
from new_ocr.ocr_run import ocr_run
import json
import logging
import kss
from flask_cors import CORS
#먼저 이 python model_server.py 입력
#그리고 새로운 터미널 열기
#ngrok http --url=sought-shrew-miserably.ngrok-free.app 8080
#고정 도메인이 있다면 위 명령어 주소 변경해서 사용하기
#없다면 ngrok.exe http 포트
#하고 나온 주소를 넣기
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

CORS(app)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

@app.route('/emotion', methods=['POST'])
def emotion():
    try:
        app.logger.info("Emotion analysis request received")
        app.logger.debug(f"Request JSON: {request.json}")
        result = emotion_run()  # 인자 제거
        return result  # 그대로 반환
    except Exception as e:
        app.logger.error(f"Error in emotion analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ocr', methods=['POST'])
def ocr():
    app.logger.info("OCR request received")
    app.logger.debug(f"Request headers: {request.headers}")
    app.logger.debug(f"Request JSON: {request.json}")
    response = ocr_run()
    app.logger.info(f"OCR response: {response}")
    return jsonify(response)

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"An unhandled exception occurred: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)