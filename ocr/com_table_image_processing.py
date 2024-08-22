import cv2
import os
import numpy as np
from PIL import Image
from ocr.com_table_config import logging as logger, config

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

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
def draw_boxes_on_image(image_path, ocr_results):
    # OpenCV로 이미지 로드
    cv_image = cv2.imread(image_path)
    
    # PIL Image로 변환
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 폰트 로드
    font = ImageFont.truetype(config['box_drawing']['font_path'], config['box_drawing']['font_size'])
    
    for line in ocr_results:
        for bbox, (text, confidence) in line:
            # 박스 그리기
            draw.polygon([tuple(point) for point in bbox], outline=tuple(config['box_drawing']['color'][::-1]), width=config['box_drawing']['thickness'])
            
            # 텍스트 추가
            text_with_conf = f"{text} ({confidence:.2f})"
            draw.text((bbox[0][0], bbox[0][1] - config['box_drawing']['font_size']), text_with_conf, font=font, fill=tuple(config['box_drawing']['color'][::-1]))
    
    # PIL Image를 OpenCV 이미지로 변환
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result_image
def draw_cells_on_image(image_path, cells):
    image = cv2.imread(image_path)
    for cell in cells:
        x1, y1, x2, y2 = cell['coordinates']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 셀 정보 표시 (선택 사항)
        cv2.putText(image, f"R{cell['row']}C{cell['col']}", (x1+5, y1+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return image

def resize_image(image_path, scale_factor=2.0):
    logger.info(f"이미지 확대 시작 (스케일 팩터: {scale_factor})")
    
    image = Image.open(image_path)
    width, height = image.size
    
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized_image = image.resize(new_size, Image.LANCZOS)
    
    # 저장 경로 설정
    base_name = os.path.basename(image_path)
    resized_path = os.path.join("resized_images", f"resized_{scale_factor}x_{base_name}")
    
    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(resized_path), exist_ok=True)
    
    resized_image.save(resized_path)
    
    logger.info(f"이미지 확대 완료: {width}x{height} -> {new_size[0]}x{new_size[1]}")
    return resized_path

def preprocess_image_for_line_detection(image_path):
    logger.info("선 검출을 위한 이미지 전처리 시작")
    
    # 이미지 로드 및 그레이스케일 변환
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 이진화
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    logger.info("선 검출을 위한 이미지 전처리 완료")
    return binary

def detect_lines(binary_image):
    logger.info("수평선 및 수직선 검출 시작")
    
    # 수평선 검출
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 수직선 검출
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # 모든 선 합치기
    all_lines = cv2.add(horizontal_lines, vertical_lines)
    
    logger.info("수평선 및 수직선 검출 완료")
    return all_lines