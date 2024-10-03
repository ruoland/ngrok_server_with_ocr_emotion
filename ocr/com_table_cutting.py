from sympy import im
from ocr.com_table_image_processing import resize_image, preprocess_image_for_line_detection, detect_lines, draw_cells_on_image
from PIL import Image, ImageEnhance
import os
import cv2
from statistics import median
import json
from ocr.com_table_config import logger
from requests import request


def crop_cells(image_path, cells):
    logger.info("셀 자르기 시작")
    
    image = Image.open(image_path)
    cropped_cells = []
    
    for row, col, (x1, y1, x2, y2) in cells:
        cell_image = image.crop((x1, y1, x2, y2))
        cropped_cells.append((row, col, cell_image))
    
    logger.info(f"셀 자르기 완료: {len(cropped_cells)}개의 셀 이미지 생성")
    return cropped_cells

def process_ocr_results(ocr_results):
    processed_results = []
    for line in ocr_results:
        for bbox, (text, confidence) in line:
            processed_results.append({
                "bbox": bbox,
                "text": text,
                "confidence": confidence
            })
    return processed_results

def postprocess_cell_image(cell_image):
    logger.info("셀 이미지 후처리 시작")
    
    # 크기 조정 (선택적)
    # cell_image = cell_image.resize((100, 100), Image.LANCZOS)
    
    # 대비 향상
    enhancer = ImageEnhance.Contrast(cell_image)
    enhanced_image = enhancer.enhance(1.5)
    
    # 선명도 향상
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    sharpened_image = enhancer.enhance(1.5)
    
    logger.info("셀 이미지 후처리 완료")
    return sharpened_image


def crop_and_save_cells(image_path, cells, output_dir):
    logger.info("셀 자르기 및 저장 시작")
    
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path)
    for cell in cells:
        if cell['is_virtual']:
            continue  # 가상 셀은 건너뛰기

        row, col = cell['row'], cell['col']
        x1, y1, x2, y2 = cell['coordinates']
        cell_image = image.crop((x1, y1, x2, y2))
        filename = f"cell_r{row}_c{col}.png"
        cell_path = os.path.join(output_dir, filename)
        cell_image.save(cell_path)
    
    logger.info(f"셀 자르기 및 저장 완료: {len(cells)}개의 셀 이미지 저장됨")
def save_image_with_boxes(image, output_path):
    cv2.imwrite(output_path, image)
def extract_cell_coordinates(lines_image):
    logger.info("셀 좌표 추출 시작")
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(lines_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 셀 좌표 추출
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # 너무 작은 셀 무시
            cells.append((x, y, x+w, y+h))
    
    # 왼쪽 위에서부터 정렬
    cells.sort(key=lambda cell: (cell[1], cell[0]))
    cells = cells[1:]  # 첫 번째 셀(전체 테이블) 제외
    
    # 중앙값 셀 높이 계산
    cell_heights = [cell[3] - cell[1] for cell in cells]
    median_cell_height = median(cell_heights)
    
    # 행과 열 정보 추가
    grid_cells = []
    prev_y = cells[0][1]
    row = -1
    col = 0
    for (x1, y1, x2, y2) in cells:
        if y1 - prev_y > median_cell_height / 2:  # 새로운 행 시작
            row += 1
            col = 0
        grid_cells.append({
            'row': row,
            'col': col,
            'coordinates': (x1, y1, x2, y2)
        })
        print(f"셀 정보 추가 중입니다. {row}, {col}, {x1,y1,x2,y2}")
        col += 1
        prev_y = y1
    
    logger.info(f"셀 좌표 추출 완료: {len(grid_cells)}개의 셀 감지")
    min_row = min(cell['row'] for cell in grid_cells)
    for cell in grid_cells:
        cell['row'] -= min_row
        cell['row_span'] = 1  # 기본값 설정
        cell['col_span'] = 1  # 기본값 설정

    grid_cells = process_merged_cells(grid_cells)
    return grid_cells, median_cell_height
def process_merged_cells(grid_cells):
    processed_cells = []
    for cell in grid_cells:
        if any(c['row'] == cell['row'] and c['col'] == cell['col'] and c != cell for c in processed_cells):
            continue  # 이미 처리된 셀 스킵
        cell['row_span'] = 1
        cell['col_span'] = 1
        for other_cell in grid_cells:
            if other_cell['row'] > cell['row'] and other_cell['col'] == cell['col']:
                if other_cell['coordinates'][1] < cell['coordinates'][3]:
                    cell['row_span'] += 1
                else:
                    break
        processed_cells.append(cell)
    return processed_cells

def process_cells_with_virtual_cells(grid_cells, median_cell_height):
    processed_cells = []
    max_row = max(cell['row'] for cell in grid_cells)
    max_col = max(cell['col'] for cell in grid_cells)
    
    # 2D 그리드 생성
    grid = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    
    # 실제 셀 배치
    for cell in grid_cells:
        row, col = cell['row'], cell['col']
        consecutive_rows = max(1, round((cell['coordinates'][3] - cell['coordinates'][1]) / median_cell_height))
        for i in range(consecutive_rows):
            if row + i <= max_row:
                grid[row + i][col] = cell

    # 가상 셀 생성 및 모든 셀 처리
    for row in range(max_row + 1):
        for col in range(max_col + 1):
            print(f'행:{row}, 열:{col} 검사 중 {grid[row][col]}, @@@@')

            if grid[row][col] is None:
                # 가상 셀 생성
                virtual_cell = {
                    'row': row,
                    'col': col,
                    'coordinates': None,  # 가상 셀은 실제 좌표가 없음
                    'consecutive_rows': 1,
                    'is_virtual': True
                }
                processed_cells.append(virtual_cell)
            else:
                cell = grid[row][col]
                if cell['row'] == row:  # 셀의 시작 행일 때만 처리
                    processed_cell = {
                        'row': cell['row'],
                        'col': cell['col'],
                        'coordinates': cell['coordinates'],
                        'consecutive_rows': max(1, round((cell['coordinates'][3] - cell['coordinates'][1]) / median_cell_height)),
                        'is_virtual': False
                    }
                    processed_cells.append(processed_cell)

    # 행과 열로 정렬
    processed_cells.sort(key=lambda x: (x['row'], x['col']))
    visualize_grid(processed_cells, max_col=max_col, max_row=max_row)
    return processed_cells

def visualize_grid(processed_cells, max_row, max_col):
    grid = [[' ' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in processed_cells:
        if cell['is_virtual']:
            grid[cell['row']][cell['col']] = 'V'
        else:
            for i in range(cell['consecutive_rows']):
                if cell['row'] + i <= max_row:
                    grid[cell['row'] + i][cell['col']] = 'C'
    
    for row in grid:
        print('|' + '|'.join(row) + '|')
        print('-' * (len(row) * 2 + 1))
def process_timetable_image(image_path, config):
    # 이미지 확대
    resized_image_path = resize_image(image_path, config['image_processing']['scale_factor'])
    
    # 전처리
    binary_image = preprocess_image_for_line_detection(resized_image_path)
    
    # 선 검출
    lines_image = detect_lines(binary_image)
    
    # 셀 좌표 추출 (0,0 셀 제외)
    cells, median_cell_height = extract_cell_coordinates(lines_image)
    
    # 가상 셀을 포함한 전체 그리드 생성
    processed_cells = process_cells_with_virtual_cells(cells, median_cell_height)
    
    # 셀 좌표를 원본 이미지 크기로 변환
    scale_factor = config['image_processing']['scale_factor']
    original_cells = []
    for cell in processed_cells:
        if not cell['is_virtual']:
            x1, y1, x2, y2 = cell['coordinates']
            original_cells.append({
                'row': cell['row'],
                'col': cell['col'],
                'coordinates': (int(x1/scale_factor), int(y1/scale_factor), 
                                int(x2/scale_factor), int(y2/scale_factor)),
                'consecutive_rows': cell['consecutive_rows'],
                'is_virtual': False
            })
        else:
            original_cells.append(cell)  # 가상 셀은 그대로 유지
    
    # 결과 이미지에 셀 그리기
    cells_to_draw = [cell for cell in original_cells if not cell.get('is_virtual', False)]
    image_with_cells = draw_cells_on_image(image_path, cells_to_draw)
    crop_and_save_cells(image_path, cells_to_draw, config['output']['cells_dir'])

    cv2.imwrite(config['output']['image_filename'], image_with_cells)
# 셀 정보를 JSON으로 저장
    cell_info = [{
        "row": cell['row'],
        "col": cell['col'],
        "coordinates": cell['coordinates'] if not cell['is_virtual'] else None,
        "consecutive_rows": cell['consecutive_rows'],
        "is_virtual": cell['is_virtual']
    } for cell in original_cells]
    save_results_to_json(cell_info, config['output']['json_filename'])
    
    return original_cells