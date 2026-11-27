# 🛒 Food Price Radar · 저렴이 레이더

공공 가격 데이터(KAMIS)로 식재료가 **해마다 반복해서 싸지는 시기**를 찾고,
생성형 AI(Gemini)가 "지금 사도 되는지"를 설명해 주는 Streamlit 대시보드.

## 파일
| 파일 | 역할 |
|---|---|
| app.py | Streamlit 화면 (탭 5개) |
| analysis.py | 통계 계산 (저가 시즌, 반복성, 평년 대비, 4주 경향, 급변 z-score) |
| llm.py | Gemini 호출 (계산된 숫자만 근거로 설명·요약·추천) |
| kamis.py | KAMIS Open-API 수집기 → prices.csv |
| secrets.toml.example | 키 설정 예시 (실제 키는 Streamlit Secrets에만) |

## 데이터 수집 (로컬 PC)
```powershell
pip install -r requirements.txt
$env:KAMIS_CERT_KEY="발급받은_인증키"
$env:KAMIS_CERT_ID="KAMIS_아이디"
python kamis.py --years 5     # prices.csv 생성 → GitHub에 업로드
```

## 배포 (Streamlit Community Cloud)
share.streamlit.io → Create app → 이 저장소, Main file `app.py` →
Advanced settings → Secrets에 `GEMINI_API_KEY = "..."` 입력 → Deploy

## 설계 포인트
LLM에게 가격을 직접 말하게 하면 숫자를 지어낸다(hallucination). 그래서
**통계는 Python이 계산 → 계산 결과만 프롬프트에 넣기 → LLM은 설명·요약·추천만** 담당하도록 분리했다.
Gemini 무료 등급을 사용하며, 접속당 AI 호출 30회 제한과 결과 캐싱으로 한도를 아낀다.
