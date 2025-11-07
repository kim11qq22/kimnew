import streamlit as st
import os
import time
import uuid
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- 상수 및 설정 ---

# 현재 시간을 기반으로 고유한 세션 ID를 생성
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.start_time = time.strftime("%Y-%m-%d %H:%M:%S")

# Gemini 모델 목록 (exp 모델 제외)
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.0-pro",
]

# LLM 시스템 프롬프트 (한국어, 고객 응대 특화)
SYSTEM_PROMPT = """
당신은 쇼핑몰 구매 과정에서 불편을 겪은 고객을 응대하는 매우 전문적이고 친절한 고객 응대 챗봇입니다.
다음 규칙을 엄격히 준수하여 응답해야 합니다:

1.  **태도:** 사용자는 쇼핑몰 구매 과정에서 겪은 불편/불만을 언급합니다. 이들의 감정에 깊이 공감하고, 정중하며 친절하고 공감 어린 존댓말투(해요체)로 응답하세요.
2.  **정보 수집 및 안내:** 사용자가 언급한 불편 사항을 구체적으로 정리하여 (무엇이/언제/어디서/어떻게) 수집하는 과정을 보여주세요. 이 정보는 고객 응대 담당자에게 전달되어 신속히 검토될 것임을 안내해야 합니다.
3.  **연락처 요청:** 담당자 확인 후 회신을 위해 대화 마지막에는 반드시 고객의 이메일 주소를 요청하세요.
4.  **연락처 거부 처리:** 만일 사용자가 연락처 제공을 명시적으로 원치 않는다면: "죄송하지만, 고객님의 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요."라고 정중히 안내하고 대화를 마무리하세요.
"""

# --- 초기화 ---

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "log_records" not in st.session_state:
    st.session_state.log_records = []
if "logging_enabled" not in st.session_state:
    st.session_state.logging_enabled = True # 기본값 설정

# --- API 클라이언트 설정 ---

def get_gemini_client():
    """API 키를 확인하고 Gemini 클라이언트를 반환합니다."""
    # 1. st.secrets에서 키를 찾습니다.
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
    else:
        # 2. secrets에 없으면, 사이드바 입력창에서 키를 가져옵니다.
        api_key = st.session_state.get('input_api_key', '')

    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            return client
        except Exception:
            st.error("잘못된 API 키 형식입니다.")
            return None
    
    # 키가 없는 경우
    return None


def log_conversation(user_text, bot_text, model_name):
    """대화 기록을 로그 레코드에 추가합니다."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_records.append({
        "session_id": st.session_state.session_id,
        "timestamp": timestamp,
        "model": model_name,
        "user_message": user_text,
        "bot_response": bot_text,
    })


# --- Streamlit UI 및 기능 ---

st.set_page_config(
    page_title="Gemini 고객 불편 접수 챗봇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛍️ Gemini 고객 불편 접수 챗봇")
st.caption("정중한 태도로 고객의 불편 사항을 접수하고 이메일을 요청합니다. (Powered by Google Gemini API)")

# --- 사이드바: 설정 및 기능 ---

with st.sidebar:
    st.header("⚙️ 설정 및 기능")

    # 1. API 키 입력 (secrets에 없는 경우)
    if 'GEMINI_API_KEY' not in st.secrets:
        st.subheader("Gemini API Key 입력")
        st.text_input(
            "API Key", 
            type="password", 
            key="input_api_key",
            placeholder="여기에 Gemini API 키를 입력하세요"
        )
    else:
        st.success("API Key가 Streamlit secrets에서 로드되었습니다.")

    # 2. 모델 선택
    st.subheader("모델 선택")
    selected_model = st.selectbox(
        "사용할 Gemini 모델",
        options=AVAILABLE_MODELS,
        index=0,
        help="사용할 기본 모델을 선택하세요. gemini-2.5-flash 권장."
    )
    
    # 3. 세션 정보
    st.subheader("세션 정보")
    st.info(f"**Session ID:** `{st.session_state.session_id}`\n\n**시작 시간:** `{st.session_state.start_time}`\n\n**현재 모델:** `{selected_model}`")

    # 4. 대화 초기화
    if st.button("🔄 대화 초기화", type="primary"):
        st.session_state.chat_history = []
        st.session_state.log_records = []
        st.rerun()

    # 5. 로깅 및 다운로드 옵션
    st.subheader("로그 및 기록 관리")
    st.checkbox(
        "CSV 자동 기록 활성화", 
        value=st.session_state.logging_enabled, 
        key="logging_enabled",
        help="모든 대화 턴을 로그 기록에 저장합니다."
    )

    log_df = pd.DataFrame(st.session_state.log_records)
    csv_data = log_df.to_csv(index=False).encode('utf-8')
    
    if len(st.session_state.log_records) > 0:
        st.download_button(
            label=f"⬇️ {len(st.session_state.log_records)}개 기록 다운로드 (.csv)",
            data=csv_data,
            file_name=f"chatbot_log_{st.session_state.session_id}.csv",
            mime="text/csv",
            help="현재 세션의 전체 대화 기록을 CSV 파일로 다운로드합니다."
        )
    else:
        st.button("⬇️ 기록 다운로드", disabled=True, help="기록된 대화가 없습니다.")


# --- 핵심 챗봇 로직 ---

client = get_gemini_client()

if not client:
    st.error("Gemini API 키를 입력하거나 `st.secrets`에 설정하여 클라이언트를 초기화해야 합니다.")
    st.stop()


def get_response(user_prompt):
    """
    API를 호출하여 응답을 받고, Rate Limit (429) 시 재시도를 수행합니다.
    최신 6턴만 Context로 사용합니다.
    """
    
    # 시스템 프롬프트와 현재까지의 대화 히스토리 (최근 6턴만) 준비
    context_history = []
    
    # History를 6턴으로 제한 (사용자 메시지 3개 + 봇 메시지 3개)
    # The list is [user, model, user, model, ...]
    # Limit to the last 6 entries (3 user turns and 3 model turns)
    limited_history = st.session_state.chat_history[-6:]
    
    # history를 API가 요구하는 형식(role, part)으로 변환
    for role, text in limited_history:
        context_history.append({
            "role": role,
            "parts": [{"text": text}]
        })

    # 마지막 사용자 프롬프트 추가
    context_history.append({
        "role": "user",
        "parts": [{"text": user_prompt}]
    })
    
    # LLM 호출 설정 (시스템 프롬프트 포함)
    config = {
        "system_instruction": SYSTEM_PROMPT,
    }

    # API 호출 및 재시도 로직
    max_retries = 3
    retry_delay = 2  # 초기 지연 시간 (초)
    
    for attempt in range(max_retries):
        try:
            # generate_content 호출 (context_history는 전체 대화가 아닌, history part + latest user prompt)
            response = client.models.generate_content(
                model=selected_model,
                contents=context_history,
                config=config,
            )
            return response.text
        
        except APIError as e:
            if "429" in str(e):
                st.warning(f"Rate Limit (429) 오류 발생. {attempt + 1}/{max_retries} 재시도 중... 다음 재시도까지 {retry_delay}초 대기.")
                time.sleep(retry_delay)
                retry_delay *= 2  # 지수 백오프
            else:
                st.error(f"API 호출 중 예측하지 못한 오류 발생: {e}")
                return "API 호출 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        except Exception as e:
             st.error(f"예기치 않은 오류가 발생했습니다: {e}")
             return "죄송합니다. 처리 중 오류가 발생했습니다."

    return "죄송합니다. API 호출 한도를 초과하여 잠시 서비스를 이용하실 수 없습니다."


# --- 대화 UI 표시 및 처리 ---

# 대화 히스토리 출력
for role, text in st.session_state.chat_history:
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

# 사용자 입력 처리
if prompt := st.chat_input("불편 사항을 말씀해 주세요."):
    
    # 1. 사용자 메시지 기록 및 표시
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. LLM 응답 생성 및 표시
    with st.chat_message("model", avatar="🤖"):
        with st.spinner("전문 담당자가 정중하게 응답을 준비하고 있어요..."):
            response_text = get_response(prompt)
            st.markdown(response_text)
            
    # 3. 응답 기록 및 로그 저장
    st.session_state.chat_history.append(("model", response_text))

    if st.session_state.logging_enabled:

        log_conversation(prompt, response_text, selected_model)
