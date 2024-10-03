import re
from new_ocr.rdtdet_log import logger

def get_day_of_week(text):
    days_kr = ['월', '화', '수', '목', '금', '토', '일']
    days_kr_full = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    days_en = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    days_en_full = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    text = text.lower()
    
    for i, day in enumerate(days_kr + days_kr_full + days_en + days_en_full):
        if day in text:
            return days_kr[i % 7]
    
    return None

def is_day_cell(cell_content):
    return get_day_of_week(cell_content) is not None


def process_day_row(timetable):
    if not timetable or len(timetable) < 2:
        logger.warning("시간표가 비어있거나 행이 충분하지 않습니다.")
        return timetable

    day_row = timetable[0]
    days = ['월', '화', '수', '목', '금', '토', '일']
    day_index = 0

    for j, cell in enumerate(day_row):
        if j == 0:  # 첫 번째 열은 건너뜁니다 (시간 정보용)
            continue
        
        if not cell['content']:
            # 내용이 없는 경우 자동으로 요일을 채웁니다
            cell['content'] = days[day_index]
            cell['day'] = days[day_index]
            day_index = (day_index + 1) % 7
        else:
            # 내용이 있는 경우 해당 내용을 요일로 간주합니다
            cell['day'] = cell['content']

        logger.info(f"열 {j}에 요일 할당: {cell['day']}")

    # 나머지 행에 대해 요일 정보를 전파합니다
    for i in range(1, len(timetable)):
        for j, cell in enumerate(timetable[i]):
            if j > 0:
                cell['day'] = timetable[0][j]['day']

    return timetable

def get_day_of_week(column):
    days = ['월', '화', '수', '목', '금', '토', '일']
    return days[(column - 1) % 7] if column > 0 else ''
