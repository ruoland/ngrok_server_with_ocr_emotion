import os
import cv2
import numpy as np
from paddleocr import PaddleOCR,draw_ocr
from new_ocr.model_detection import initialize_model
from new_ocr.rdtdet_merge import merge_ocr_results
from new_ocr.rdtdet_io import save_cropped_images, save_json
from new_ocr.rdtdet_process import process_image
from new_ocr.rdtdet_analyze import analyze_and_create_timetable
from new_ocr.rdtdet_display import display_results, print_merged_ocr_results
import uuid
import requests

# 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 모델 초기화
ocr = PaddleOCR(use_angle_cls=False, lang='korean')
rtmdet = initialize_model()
IMAGE_THRESHOLD = 0.5

def ocr_run():
    try:
        from flask import request
        data = request.json
        if not data or 'imageUrl' not in data:
            return {'error': 'No image URL provided'}, 400

        image_url = data['imageUrl']
        
        # 이미지 다운로드
        response = requests.get(image_url)
        if response.status_code != 200:
            return {'error': 'Failed to download image'}, 400
        
        # 이미지를 NumPy 배열로 변환
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            return {'error': 'Failed to decode image'}, 400

        # 이미지 처리
        original_img, detected_objects, ocr_results = process_image(img, rtmdet, ocr, IMAGE_THRESHOLD)

        # OCR 결과 병합
        detected_cells = [obj for obj in detected_objects if obj['label'] == 0]
        merged_ocr_results = merge_ocr_results(ocr_results, detected_cells)

        # 시간표 분석 및 생성
        timetable, header_row, header_col = analyze_and_create_timetable(detected_objects, merged_ocr_results)

        # 시간표 엔트리 생성
        timetable_entries = []
        for i, row in enumerate(timetable):
            for j, cell in enumerate(row):
                if cell['content']:
                    entry = {
                        "id": str(uuid.uuid4()),
                        "day_of_week": timetable[0][j].get('day', ''),
                        "period": timetable[i][0].get('time', str(i)),
                        "start_time": cell.get('start_time', ''),
                        "end_time": cell.get('end_time', ''),
                        "subject": cell['content'],
                        "row": cell['row'],
                        "column": j,
                        "consecutive_classes": cell.get('consecutive_classes', 1),
                        "cell_type": cell['type']
                    }
                    timetable_entries.append(entry)

        return timetable_entries

    except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {'error': str(e)}, 500
