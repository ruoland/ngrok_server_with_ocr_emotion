import logging
import yaml
import re

# 로깅 레벨을 INFO로 명시적으로 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 로거의 레벨이 제대로 설정되었는지 확인

logger.setLevel(logging.INFO)

def load_config(config_path='./ocr/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()

# 로깅이 제대로 설정되었는지 확인
