import streamlit as st
import os
import time
import uuid
import pandas as pd
import google.generativeai as genai  # ✅ 올바른 import 수정됨
from google.api_core import exceptions as google_exceptions

# --- 상수 및 설정 ---

# 세션 ID 및 시작 시간 초기화
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.start_time = time.strftime("%Y-%m-%d %H:%M:%S")

# 사용 가능한 모델 목록
AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 쇼핑몰 구매 과정에서 불편을 겪은 고객을 응대하는 매우 전문적이고 친절한 고객 응대 챗봇입니다.
다음 규칙을 엄격히 준수하여 응답해야 합니다:

1. **태도:** 사용자는 쇼핑몰 구매 과정에서 겪은 불편/불만을 언급합니다. 이들의 감정에 공감하고 정중한 존댓말로 응답하세요.
2. **정보 수집 및 안내:** 사용자가 언급한 불편 사항을 구체적으로 정리하여 담당자에게 전달된다고 안내하세요.
3. **연락처 요청:** 담당자 회신을 위해 이메일 주소를 요청하세요.
4. **연락처 거부 처리:** 사용자가 이메일 제공을 거부하면 정중하게 안내하고 대화를 종료하세요.
"""

# --- 세션 상태 초기화 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "log_records" not in st.session_state:
    st.session_state.log_records = []
if "logging_enabled" not in st.session_state:
    st.session_state.logging_enabled = True

# --- Streamlit 페이지 설정 ---
st.set_page_config(
    page_title="Gemini 고객 불편 접수 챗봇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛍️ Gemini 고객 불편 접수 챗봇")
st.caption("정중한 태도로 고객의 불편 사항을 접수하고 이메일을 요청합니다. (Powered by Google Gemini API)")

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")

    # API 키 입력
    api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="여기에 API 키를 입력하세요")
    if not api_key:
        st.warning("Gemini API 키를 입력해야 챗봇을 사용할 수 있습니다.")
    else:
        genai.configure(api_key=api_key)

    # 모델 선택
    selected_model = st.selectbox("모델 선택", AVAILABLE_MODELS, index=0)

    # 로그 다운로드
    if st.session_state.log_records:
        log_df = pd.DataFrame(st.session_state.log_records)
        st.download_button(
            label="⬇️ 대화 기록 다운로드 (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name=f"chat_log_{st.session_state.session_id}.csv",
            mime="text/csv"
        )

    if st.button("🔄 대화 초기화"):
        st.session_state.chat_history.c
