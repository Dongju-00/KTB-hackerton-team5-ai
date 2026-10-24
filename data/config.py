"""
API 키 로드, HTTP 세션 설정

경로 상수는 전부 PROJECT_ROOT(이 파일 기준) 아래로 잡으므로 CWD와 무관하다.
다만 import는 프로젝트 루트 기준 절대 import라, 실행은 ai/ 에서 한다.

check_keys() -> API 키들이 제대로 들어가 있는지 확인하는 함수
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()  # 환경변수 파일 불러오기

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 이 파일은 ai/data/config.py 이므로 두 단계 위가 프로젝트 루트(ai/)다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 캐시 폴더. CWD가 아니라 이 파일 기준이라, 어느 위치에서 실행해도 같은 자리를 가리킨다.
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 병원 개폐업/이전이 반영되어야 하므로 응답은 캐싱하지 않는다.
session = requests.Session()


def check_keys():
    """
    키가 전부 제대로 들어가 있는지 확인하고, 하나라도 누락됐다면 누락된 키의 이름과 함께 오류를 띄우는 함수.

    런타임 중간이 아니라 시작 시점에 실패시키기 위한 것이다.
    """
    missing_keys = [
        name
        for name, key in [
            ("OPENAI_API_KEY", OPENAI_API_KEY),
        ]
        if not key
    ]

    if missing_keys:
        raise RuntimeError(f"키 누락: {', '.join(missing_keys)}.")

