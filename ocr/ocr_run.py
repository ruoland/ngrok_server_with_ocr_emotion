from flask import request, jsonify
import requests
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ocr.com_table_config import load_config, logger
from ocr.com_table_cutting import process_timetable_image
import tempfile
import os
import traceback
import logging
from flask import request, jsonify
import re
import traceback
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def initialize_ocr():
    return PaddleOCR(use_angle_cls=True, lang='korean')

def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def ocr_run():
    try:
        logger.info("OCR 처리 시작")
        data = request.json
        if not data or 'imageUrl' not in data:
            logger.error("이미지 URL이 제공되지 않았습니다.")
            return jsonify({'error': 'No image URL provided in the request'}), 400
        
        image_url = data['imageUrl']
        logger.info(f"처리할 이미지 URL: {image_url}")
        
        # 이미지 URL에서 다운로드
        img = download_image(image_url)
        if img is None:
            return jsonify({'error': 'Failed to download image from URL'}), 400

        # 임시 파일로 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            cv2.imwrite(temp_file.name, img)
            temp_file_path = temp_file.name

        try:
            config = load_config()
            ocr = initialize_ocr()

            # 이미지 처리 및 OCR 로직
            processed_cells = process_timetable_image(temp_file_path, config)
            logger.info(f"Processed cells: {processed_cells}")

            processed_cells = ensure_span_attributes(processed_cells)
            processed_cells = create_virtual_cells(processed_cells)
            processed_cells = sort_cells(processed_cells)
            logger.info(f"Cells after processing: {processed_cells}")

            time_cells, day_cells, content_cells = process_cells(processed_cells, img, ocr)
            logger.info(f"Time cells: {time_cells}")
            logger.info(f"Day cells: {day_cells}")
            logger.info(f"Content cells: {content_cells}")

            content_cells = assign_days_and_times(content_cells, day_cells, time_cells)
            logger.info(f"Content cells after assigning days and times: {content_cells}")

            supabase_data = prepare_for_supabase(content_cells)
            logger.info(f"Supabase data: {supabase_data}")
            
            result = {"message": "OCR 처리 완료", "data": supabase_data}
            logger.info("OCR 처리 완료")
            return jsonify(result), 200

        finally:
            # 임시 파일 삭제
            os.unlink(temp_file_path)
    except Exception as e:
           logger.error(f"OCR 처리 중 오류 발생: {str(e)}")
           logger.error(traceback.format_exc())
           return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

def initialize_ocr():
    return PaddleOCR(use_angle_cls=True, lang='korean')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return [gray, denoised, binary, morph]

def perform_ocr(cell_image, ocr):
    preprocessed_images = preprocess_image(cell_image)
    best_result = {"text": "", "confidence": 0}

    for img in preprocessed_images:
        result = ocr.ocr(img, cls=False)
        if result and result[0]:
            text = "\n".join([line[1][0] for line in result[0]])
            confidence = sum([line[1][1] for line in result[0]]) / len(result[0])
            
            if confidence > best_result["confidence"]:
                best_result = {"text": text, "confidence": confidence}

    return best_result


def is_day_of_week(text):
    day_pattern = r"^(월|화|수|목|금|토|일|월요일|화요일|수요일|목요일|금요일|토요일|일요일)$"
    day_match = re.match(day_pattern, text)
    return bool(day_match)

def is_time_cell(text, col, all_cells):
    time_patterns = [
        r'\d{1,2}[:：]\d{2}',  # HH:MM or H:MM
        r'\d{1,2}[:：]\d{2}[-~]\d{1,2}[:：]\d{2}',  # HH:MM-HH:MM or HH:MM~HH:MM
        r'\d+교시'  # N교시
    ]
    
    if any(re.search(pattern, text) for pattern in time_patterns) or '교시' in text:
        return True
    
    same_col_cells = [cell for cell in all_cells if cell['col'] == col]
    time_cells_in_col = sum(1 for cell in same_col_cells if 
                            any(re.search(pattern, cell.get('text', '')) for pattern in time_patterns) or 
                            '교시' in cell.get('text', ''))
    return time_cells_in_col / len(same_col_cells) >= 0.5 if same_col_cells else False

def process_cells(processed_cells, crop_img, ocr):
    time_cells = []
    day_cells = []
    content_cells = []
    
    for i, cell in enumerate(processed_cells):
        default_cell = create_default_cell(cell['row'], cell['col'])
        default_cell.update(cell)  # 기존 정보 유지
        
        if not cell['is_virtual']:
            x1, y1, x2, y2 = cell['coordinates']
            cell_image = crop_img[y1:y2, x1:x2]
            cell_image = cv2.resize(cell_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            result = perform_ocr(cell_image, ocr)
            default_cell['text'] = result['text']
            default_cell['confidence'] = result['confidence']

        # 요일 셀 처리
        if default_cell['row'] == 0 and is_day_of_week(default_cell['text']):
            day_cells.append(default_cell)
        # 교시 셀 처리
        elif default_cell['col'] == 0 and is_time_cell(default_cell['text'], default_cell['col'], processed_cells):
            period, start_time, end_time = extract_time_info(default_cell['text'])
            default_cell['period'] = period
            default_cell['start_time'] = start_time
            default_cell['end_time'] = end_time
            time_cells.append(default_cell)
        else:
            content_cells.append(default_cell)

        processed_cells[i] = default_cell  # 여기서 업데이트된 셀 정보를 원래 리스트에 저장

    return time_cells, day_cells, content_cells
def create_default_cell(row, col, is_virtual=False):
    return {
        'row': row,
        'col': col,
        'is_virtual': is_virtual,
        'row_span': 1,
        'col_span': 1,
        'coordinates': None if is_virtual else (),
        'consecutive_rows': 1,
        'start_row': row,
        'end_row': row,
        'start_time': "",
        'end_time': "",
        'period': f"{row + 1}교시",
        'day': "",
        'text': ""
    }
def sort_cells(cells):
    return sorted(cells, key=lambda c: (c['row'], c['col']))
def create_virtual_cells(processed_cells):
    max_row = max(cell['row'] + cell.get('row_span', 1) for cell in processed_cells)
    max_col = max(cell['col'] + cell.get('col_span', 1) for cell in processed_cells)
    virtual_cells = []
    
    for row in range(max_row):
        for col in range(max_col):
            if not any(cell['row'] <= row < cell['row'] + cell.get('row_span', 1) and 
                       cell['col'] <= col < cell['col'] + cell.get('col_span', 1) for cell in processed_cells):
                virtual_cells.append(create_default_cell(row, col, is_virtual=True))
    return sort_cells(processed_cells + virtual_cells)
def assign_days_and_times(content_cells, day_cells, time_cells):
    time_slots = sorted([
        (cell['row'], 
         cell['period'] if 'period' in cell else f"{cell['row']+1}교시", 
         cell.get('start_time', ''), 
         cell.get('end_time', '')) 
        for cell in time_cells if 'period' in cell
    ])
    assigned_cells = []

    if not time_slots:
        logger.warning("시간 슬롯이 없습니다. 모든 셀에 기본 시간 정보를 할당합니다.")
        for cell in content_cells:
            cell['start_row'] = cell['row']
            cell['end_row'] = cell['row']
            cell['start_time'] = ""
            cell['end_time'] = ""
            cell['period'] = f"{cell['row'] + 1}교시"
            cell['day'] = next((day_cell['text'] for day_cell in day_cells if day_cell['col'] == cell['col']), '')
            assigned_cells.append(cell)
        return assigned_cells

    for cell in content_cells:
        start_slot = next((i for i, (row, _, _, _) in enumerate(time_slots) if row >= cell['row']), len(time_slots) - 1)
        end_slot = min(start_slot + cell['consecutive_rows'] - 1, len(time_slots) - 1)
        print(cell['is_virtual'], " 가상셀 확인..")
        cell['day'] = next((day_cell['text'] for day_cell in day_cells if day_cell['col'] == cell['col']), '')
        if start_slot < len(time_slots):
            cell['start_row'] = time_slots[start_slot][0]
            cell['end_row'] = time_slots[end_slot][0]
            cell['start_time'] = time_slots[start_slot][2]
            # 여러 셀을 차지하는 경우, 마지막 셀의 end_time을 사용
            if end_slot > start_slot:
                cell['end_time'] = time_slots[end_slot][3]
            elif cell['start_time']:
                # 단일 셀인 경우, 시작 시간에 기본 수업 시간(예: 50분)을 더함
                start_hour, start_minute = map(int, cell['start_time'].split(':'))
   
                cell['end_time'] = ''
            
            cell['period'] = f"{time_slots[start_slot][1]}"
            
        else:
            logger.warning(f"셀 {cell['row']}행 {cell['col']}열에 대한 시간 슬롯을 찾을 수 없습니다. 기본값을 할당합니다.")
            cell['start_row'] = cell['row']
            cell['end_row'] = cell['row']
            cell['start_time'] = ""
            cell['end_time'] = ""
            cell['period'] = f"{cell['row'] + 1}교시"
            cell['day_of_week'] = ''


        assigned_cells.append(cell)

    return assigned_cells

def prepare_for_supabase(content_cells):
    return [{
        "day_of_week": cell.get('day', ''),
        "period": cell['period'],
        "start_time": cell['start_time'],
        "end_time": cell['end_time'],
        "subject": cell['text'],
        "row": cell['row'],
        "column": cell['col'],
        "consecutive_classes": cell['consecutive_rows']
    } for cell in content_cells]


def calculate_period(time):
    hour = int(time.split(':')[0])
    return str(hour - 8) if hour >= 9 else "1"
def correct_ocr_errors(text):
    text = re.sub(r'(\d+)[oO]', r'\g<1>0', text)
    text = re.sub(r'(\d+)[lI]', r'\g<1>1', text)
    return text

def calculate_end_time(start_time):
    hour, minute = map(int, start_time.split(':'))
    end_minute = minute + 45
    end_hour = hour + (end_minute // 60)
    end_minute %= 60
    return f"{end_hour:02d}:{end_minute:02d}"

def extract_time_info(text):
    text = correct_ocr_errors(text)

    period = extract_period(text)  # 교시 정보 추출
    start_time, end_time = extract_time(text)  # 시간 정보 추출

    logger.warn(f"교시: {period}, 시작 시간:{start_time}, 끝나는 시간:{end_time}")
    return period, start_time, end_time

def extract_period(text):
    """교시 정보를 추출합니다."""
    period_pattern = r"(\d+교시|\S+)"  # 교시 패턴 (필요에 따라 수정 가능)
    period_match = re.search(period_pattern, text)
    return period_match.group(1) if period_match else ""

def extract_time(text):
    """시간 정보만 추출합니다. (4자리 시간 형식 지원)"""
    pattern = r"(\d{1,2}[:：]\d{2}(?:\s*[-~]\s*\d{1,2}[:：]\d{2})?)"
    match = re.search(pattern, text)
    
    if match:
        time_part = match.group(1)
        if "-" in time_part or "~" in time_part:
            start_time, end_time = map(str.strip, time_part.split("-" if "-" in time_part else "~"))
        else:
            start_time = time_part.strip()
            end_time = ""
        return start_time, end_time
    else:
        return "", ""

def extract_day(text):
    """요일 정보를 추출합니다."""
    day_pattern = r"(월|화|수|목|금|토|일|월요일|화요일|수요일|목요일|금요일|토요일|일요일)"
    day_match = re.search(day_pattern, text)
    return day_match.group(1) if day_match else ""
def ensure_span_attributes(cells):
    for cell in cells:
        if 'row_span' not in cell:
            cell['row_span'] = 1
        if 'col_span' not in cell:
            cell['col_span'] = 1
    return cells
def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
def main():
    config = load_config()
    ocr = initialize_ocr()

    if len(sys.argv) > 1 and sys.argv[1].startswith('http'):
        # URL에서 이미지 다운로드
        image_url = sys.argv[1]
        logger.info(f"Downloading image from URL: {image_url}")
        img = download_image(image_url)
        if img is None:
            logger.error(f"Failed to download image from {image_url}")
            return
        
        # 임시 파일로 저장
        temp_file_path = 'temp_image.jpg'
        cv2.imwrite(temp_file_path, img)
        file_path = temp_file_path
    elif len(sys.argv) > 1:
        # 로컬 파일 경로 사용
        file_path = sys.argv[1]
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"{file_path}에서 이미지를 불러오지 못했습니다.")
            return
    else:
        print('인식할 이미지 파일 경로 또는 URL을 입력해주세요.')
        print('python main.py [파일경로 또는 URL]')
        return

   # 이미지 처리 및 OCR 로직
    processed_cells = process_timetable_image(file_path, config)
    processed_cells = ensure_span_attributes(processed_cells)
    processed_cells = create_virtual_cells(processed_cells)
    processed_cells = sort_cells(processed_cells)

    time_cells, day_cells, content_cells = process_cells(processed_cells, img, ocr)
    content_cells = assign_days_and_times(content_cells, day_cells, time_cells)
    supabase_data = prepare_for_supabase(content_cells)
    
    save_results_to_json(supabase_data, config['output']['json_filename'])
    
    logger.info(f"{len(supabase_data)} 의 셀을 작업 완료 했습니다. output 폴더를 참조하세요.")

    # 임시 파일 삭제 (URL에서 다운로드한 경우)
    if file_path == 'temp_image.jpg':
        import os
        os.remove(file_path)