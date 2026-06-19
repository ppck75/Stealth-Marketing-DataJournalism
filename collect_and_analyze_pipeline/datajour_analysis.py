"""Basic analysis utilities for crawl_data_{category}.csv files.

This script intentionally does not define advertising-suspicion indicators.
Add your own indicator functions/prompts in the marked sections when ready.

Examples:
    python datajour_analysis.py --mode audit
    DATA_JOUR_BASE_DIR=/path/to/data python datajour_analysis.py --mode audit
    python datajour_analysis.py --mode export-llm-input --input-csv crawl_data_비염.csv --limit 20
    python datajour_analysis.py --mode llm --input-csv crawl_data_비염.csv --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE_DIR = Path(os.getenv("DATA_JOUR_BASE_DIR", ".")).expanduser().resolve()
OUTPUT_DIR = BASE_DIR / "analysis_outputs"
CHAT_KHU_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
CHAT_KHU_CHAT_COMPLETIONS_URL = f"{CHAT_KHU_BASE_URL}/chat/completions"
CHAT_KHU_MODEL = "gemini-3.1-flash-lite"
DEFAULT_LLM_SLEEP_SEC = float(os.getenv("LLM_SLEEP_SEC", "0.5"))
CREDIT_EXHAUSTION_MESSAGE = (
    "API 크레딧/한도/인증 오류가 감지되어 분석을 중단합니다. "
    "지금까지 성공한 결과는 JSONL/CSV에 저장되어 있습니다. "
    "크레딧 충전 후 같은 명령어로 다시 실행하면 성공 저장된 게시글은 건너뛰고 남은 게시글부터 이어서 분석합니다."
)

CATEGORIES = ["비염", "탈모", "피부미용", "키성장", "다이어트"]
DATA_FILES = {category: BASE_DIR / f"crawl_data_{category}.csv" for category in CATEGORIES}

REQUIRED_COLUMNS = [
    "source",
    "title",
    "description",
    "link",
    "postdate",
    "site_name",
    "site_url",
    "core_keyword",
    "search_keywords",
    "matched_keyword_count",
    "matched_keywords",
    "full_text",
    "full_text_status",
    "comment_count_collected",
    "comment_fetch_status",
    "comments_file_key",
]

LLM_TRACKING_COLUMNS = [
    "post_id",
    "core_keyword",
    "source",
    "title",
    "description",
    "link",
    "source_file",
    "search_keywords",
    "matched_keywords",
    "comment_count_collected",
    "comment_fetch_status",
    "comments_file_key",
]

LLM_PROMPT_METADATA_COLUMNS = [
    "core_keyword",
    "source",
    "search_keywords",
    "matched_keywords",
    "comment_count_collected",
]

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
아래의 [Role], [Step 1], [Step 2], [출력 형식]은 모두 반드시 따라야 하는 분석 지침의 섹션 제목입니다.
분석 대상 텍스트는 맨 아래 [분석 대상 텍스트]에 제공됩니다.
반드시 Step 1을 먼저 적용하고, Step 1에서 정상 광고로 판정되지 않은 경우에만 Step 2를 적용하십시오.

# [Role]
당신은 온라인 커뮤니티, 블로그 게시글의 텍스트를 분석하여 '기만적 스텔스 마케팅(허위 바이럴)' 여부를 판별하는 전문 AI 에이전트입니다. 주어진 텍스트를 분석할 때 반드시 아래의 2단계 파이프라인을 엄격하게 준수하여 평가하십시오.

---
# [Step 1: 합법적 광고 사전 검증 (Gatekeeper)]
본격적인 스텔스 마케팅 분석에 앞서, 해당 텍스트가 표시광고법을 준수한 '정상 광고'인지 먼저 필터링합니다. 아래 2가지 요건을 **모두** 충족하는지 확인하십시오.

1. **위치의 명확성:** 대가성 표기(원고료, 제품 제공 등)가 소비자가 즉시 인지할 수 있는 글의 제목이나 본문 최상단에 명시되어 있는가?
2. **표현의 확정성:** "지급받을 수 있음"과 같은 조건부/회피성 표현이 아니라, "지원받았습니다", "소정의 원고료를 받았습니다" 등 확정적인 문구인가?


**[Step 1 판정 규칙]**
* 위 요건 2가지를 모두 충족한다면, 해당 게시물은 **[정상 광고]**입니다. 
* **결과가 [정상 광고]일 경우, 즉시 평가를 종료하고 Step 2를 진행하지 마십시오.** (출력 양식의 '정상 광고' 포맷으로만 답변할 것)
* 위 요건 중 하나라도 누락되었거나 대가성 표기가 전혀 없다면, 기만성이 의심되므로 **[Step 2]**로 넘어가십시오.

---
# [Step 2: 스텔스 마케팅 의심도 심층 평가 (Suspicion Scoring)]
Step 1을 통과하지 못한 텍스트에 대해, 아래 13가지 판별 기준(C1~C13)의 충족 여부를 분석하십시오.

**[평가 기준: 13가지 체크리스트]**
* C1. 후기/체험담 형식을 사용하는가? (세부 예시: '직접 써봤는데요', '솔직 후기', '체험단 리뷰')
* C2. SNS 공유를 유도하는 표현이 있는가? (세부 예시: '저장 필수', '공유해 주세요', '태그 이벤트', '주변에 알려주세요')
* C3. 소비자 추천처럼 보이는 표현이 있는가? (세부 예시: '강추', '이거 진짜 좋음', '안 사면 후회해요', '주변에도 알려줬어요' 등)
* C4. 브랜드 홍보 의도가 숨겨져 있는가? (세부 예시: 브랜드명 간접 언급, 정보성 글 안에 제품 링크나 할인코드를 자연스럽게 삽입)
* C5. 감정적 호소 표현이 포함되어 있는가? (세부 예시: '눈물남', '완전 공감', '나만 알기 아까워서', '써보고 소름 돋았음')
* C6. 참여·이벤트형 구조가 포함되어 있는가? (세부 예시: 챌린지, 인증샷 이벤트, '리그램하면 추첨', 댓글 참여 유도)
* C7. 협찬/광고비 수수 사실이 불분명한가? (세부 예시: '#협찬', ‘#광고’ 없이 제품 소개)
* C8. 인플루언서/파워블로거 형식으로 작성됐는가? (세부 예시: 팔로워 수 강조, 전문가 포지셔닝, 정기적 제품 소개 콘텐츠)
* C9. 일방적 긍정 표현이 과도하게 반복되는가? (세부 예시: 단점 언급이 없고 극찬 반복, '완벽해요', '단점을 모르겠어요')
* C10. 타 업체와의 비교를 통해 우월성을 강조하는가? (세부 예시: "예전에 쓰던 다른 곳들은 다 문제가 많았는데 여기는 다르다", 타사의 단점을 구체적으로 지적하며 현재 업체의 장점을 부각하는 구조)
* C11. 언어 사용이 부자연스럽거나 마케팅 용어가 빈번한가? (세부 예시: "최고의 서비스", "완벽한 시스템" 같은 정제되고 단정적인 마케팅 단어 남발, 오탈자나 자연스러운 실수가 없는 완벽한 문장, 실제 경험에서 나오는 구체적 디테일의 결여)
* C12. 객관적 근거 없는 허위·과장 표현이 포함되어 있는가? (세부 예시: '100% 효과 보장', '부작용 전혀 없음' 등 객관적 검증이 불가능한 단정적 표현, 자발적 리뷰를 가장하여 효능을 부풀린 확정적 문구)
* C13. 조건부·불확정적 표현으로 대가성을 흐리는가? (세부 예시: "일정 수수료를 지급받을 수 있음", "판매에 따라 커미션이 발생할 수도 있음" 등 대가 수수 여부를 확정하지 않고 모호하고 빠져나갈 구멍을 두는 표현 사용)

**[Step 2 판정 규칙 및 위험도 산출]**
발견된 항목의 총개수에 따라 아래 기준으로 위험도를 평가합니다.
* **검토우선:** 10~13개 충족
* **검토필요:** 5~9개 충족
* **검토낮음:** 0~4개 충족

---
# [출력 형식 (JSON)]
반드시 아래 JSON 스키마를 엄격하게 준수하여 출력하십시오. 그 외의 설명이나 텍스트는 절대 포함하지 마십시오.
{{
    "합법적_광고_사전_검증_여부": "[정상 광고 / 기만광고 의심]",
    "위험도": "[정상 광고 / High / Medium / Low]",
    "리스트_내_체크_갯수": [발견된 체크리스트 항목의 정수 개수, Step 1 통과 시 0],
    "체크된_항목_리스트": ["[C1, C4 등 체크된 항목 번호]", Step 1 통과 시 빈 리스트 []]
}}

# [분석 대상 텍스트]
{text}
""".strip()


@dataclass
class AnalysisConfig:
    input_csv: Path | None
    categories: list[str]
    mode: str
    limit: int | None
    sample_per_category: int | None
    max_text_chars: int
    prompt_file: Path | None
    model: str
    sleep_sec: float
    resume: bool
    max_consecutive_errors: int


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")


def make_post_id(row: pd.Series) -> str:
    category = str(row.get("core_keyword", "") or row.get("analysis_category", "")).strip()
    link = str(row.get("link", "")).strip()
    return f"{category}::{link}"


def normalize_count(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", "0"), errors="coerce").fillna(0).astype(int)


def infer_category_from_path(path: Path) -> str:
    stem = path.stem
    prefix = "crawl_data_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def prepare_loaded_posts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["post_id"] = df.apply(make_post_id, axis=1)
    df["full_text_length_calc"] = df.get("full_text", pd.Series([""] * len(df))).astype(str).str.len()
    df["comment_count_int"] = normalize_count(
        df.get("comment_count_collected", pd.Series([0] * len(df)))
    )
    df["matched_keyword_count_int"] = normalize_count(
        df.get("matched_keyword_count", pd.Series([0] * len(df)))
    )
    return df


def load_posts_from_input_csv(input_csv: Path) -> pd.DataFrame:
    path = input_csv.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"입력 CSV 파일이 없습니다: {path}")

    df = read_csv(path)
    inferred_category = infer_category_from_path(path)
    if "core_keyword" not in df.columns or df["core_keyword"].astype(str).str.strip().eq("").all():
        df["core_keyword"] = inferred_category
    df["analysis_category"] = inferred_category
    df["source_file"] = str(path)
    return prepare_loaded_posts(df)


def load_posts_from_categories(categories: Iterable[str]) -> pd.DataFrame:
    frames = []
    for category in categories:
        path = DATA_FILES.get(category, BASE_DIR / f"crawl_data_{category}.csv")
        if not path.exists():
            raise FileNotFoundError(f"파일이 없습니다: {path}")

        df = read_csv(path)
        df["analysis_category"] = category
        df["source_file"] = str(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return prepare_loaded_posts(combined)


def load_posts(config: AnalysisConfig) -> pd.DataFrame:
    if config.input_csv is not None:
        return load_posts_from_input_csv(config.input_csv)
    return load_posts_from_categories(config.categories)


def validate_posts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_file, part in df.groupby("source_file", dropna=False):
        missing = [col for col in REQUIRED_COLUMNS if col not in part.columns]
        category_values = sorted(
            value for value in part.get("core_keyword", pd.Series(dtype=str)).astype(str).unique() if value
        )
        link_dup_count = int(part.duplicated(subset=["link"]).sum()) if "link" in part.columns else None
        post_id_dup_count = int(part.duplicated(subset=["post_id"]).sum()) if "post_id" in part.columns else None
        rows.append(
            {
                "source_file": source_file,
                "rows": len(part),
                "missing_columns": ", ".join(missing),
                "core_keyword_values": ", ".join(category_values),
                "duplicate_link_drop_count": link_dup_count,
                "duplicate_post_id_drop_count": post_id_dup_count,
                "full_text_success_rows": int((part.get("full_text_status", "") == "success").sum())
                if "full_text_status" in part.columns
                else None,
                "nonempty_full_text_rows": int(
                    part.get("full_text", pd.Series(dtype=str)).astype(str).str.strip().ne("").sum()
                )
                if "full_text" in part.columns
                else None,
            }
        )
    return pd.DataFrame(rows)


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("core_keyword", dropna=False)
    return grouped.agg(
        rows=("link", "size"),
        unique_urls=("link", "nunique"),
        blog_rows=("source", lambda s: int((s == "blog").sum())),
        cafe_rows=("source", lambda s: int((s == "cafearticle").sum())),
        full_text_success=("full_text_status", lambda s: int((s == "success").sum())),
        nonempty_full_text=("full_text", lambda s: int(s.astype(str).str.strip().ne("").sum())),
        avg_full_text_length=("full_text_length_calc", "mean"),
        min_full_text_length=("full_text_length_calc", "min"),
        max_full_text_length=("full_text_length_calc", "max"),
        posts_with_comments=("comment_count_int", lambda s: int((s > 0).sum())),
        total_collected_comments=("comment_count_int", "sum"),
        avg_comments_per_post=("comment_count_int", "mean"),
    ).reset_index()


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["core_keyword", "source"], dropna=False)
    return grouped.agg(
        rows=("link", "size"),
        unique_urls=("link", "nunique"),
        full_text_success=("full_text_status", lambda s: int((s == "success").sum())),
        posts_with_comments=("comment_count_int", lambda s: int((s > 0).sum())),
        total_collected_comments=("comment_count_int", "sum"),
    ).reset_index()


def status_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=["core_keyword", column, "rows"])
    return (
        df.groupby(["core_keyword", column], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["core_keyword", "rows"], ascending=[True, False])
    )


def select_llm_input(df: pd.DataFrame, config: AnalysisConfig, apply_limits: bool = True) -> pd.DataFrame:
    selected = df[df.get("full_text", pd.Series([""] * len(df))).astype(str).str.strip().ne("")].copy()
    selected = selected.sort_values(["core_keyword", "source", "post_id"]).reset_index(drop=True)

    if apply_limits and config.sample_per_category:
        selected = (
            selected.groupby("core_keyword", group_keys=False)
            .head(config.sample_per_category)
            .reset_index(drop=True)
        )
    if apply_limits and config.limit:
        selected = selected.head(config.limit).reset_index(drop=True)

    selected["full_text_for_llm"] = selected["full_text"].astype(str).str.slice(0, config.max_text_chars)
    columns = [col for col in LLM_TRACKING_COLUMNS if col in selected.columns]
    columns.append("full_text_for_llm")
    return selected[columns]


# -----------------------------------------------------------------------------
# Add your own advertising-suspicion indicators below.
# Keep this function explicit so indicator definitions remain researcher-owned.
# -----------------------------------------------------------------------------
def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for user-defined indicators.

    Do not add project indicators here unless they are explicitly defined by the
    research team. The current implementation returns the data unchanged.
    """
    return df


def get_output_stem(config: AnalysisConfig) -> str:
    if config.input_csv is not None:
        return config.input_csv.expanduser().resolve().stem
    if len(config.categories) == 1:
        return f"crawl_data_{config.categories[0]}"
    return "all_categories"


def save_audit_outputs(df: pd.DataFrame, config: AnalysisConfig) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_stem = get_output_stem(config)

    validation = validate_posts(df)
    category = category_summary(df)
    source = source_summary(df)
    full_text_status = status_summary(df, "full_text_status")
    comment_status = status_summary(df, "comment_fetch_status")

    outputs = {
        "data_validation_summary.csv": validation,
        "category_summary.csv": category,
        "source_summary.csv": source,
        "full_text_status_summary.csv": full_text_status,
        "comment_fetch_status_summary.csv": comment_status,
    }

    for filename, output_df in outputs.items():
        path = OUTPUT_DIR / f"{output_stem}_{filename}"
        output_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {path}")

    print("\n주제별 요약")
    print(category.to_string(index=False))


def export_llm_input(df: pd.DataFrame, config: AnalysisConfig) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_stem = get_output_stem(config)
    llm_input = select_llm_input(df, config)
    path = OUTPUT_DIR / f"{output_stem}_llm_input_posts.csv"
    llm_input.to_csv(path, index=False, encoding="utf-8-sig")

    jsonl_path = OUTPUT_DIR / f"{output_stem}_llm_input_posts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in llm_input.to_dict("records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"LLM 입력 CSV 저장 완료: {path}")
    print(f"LLM 입력 JSONL 저장 완료: {jsonl_path}")
    print(f"LLM 입력 행 수: {len(llm_input):,}")
    return path


def load_prompt_template(prompt_file: Path | None) -> str:
    if prompt_file is None:
        return DEFAULT_SYSTEM_PROMPT_TEMPLATE
    if not prompt_file.exists():
        raise FileNotFoundError(f"프롬프트 파일이 없습니다: {prompt_file}")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if "{text}" not in prompt:
        raise ValueError("프롬프트 파일에는 분석 텍스트가 들어갈 {text} placeholder가 필요합니다.")
    return prompt


def build_llm_text(row: pd.Series, max_text_chars: int) -> str:
    metadata = {
        column: row.get(column, "")
        for column in LLM_PROMPT_METADATA_COLUMNS
    }
    title = str(row.get("title", ""))
    description = str(row.get("description", ""))
    full_text = str(row.get("full_text_for_llm", row.get("full_text", "")))[:max_text_chars]
    return (
        f"[메타데이터]\n{json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"[제목]\n{title}\n\n"
        f"[검색 요약문]\n{description}\n\n"
        f"[본문]\n{full_text}"
    )


def parse_json_response(raw: str) -> dict:
    raw = str(raw).strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def get_chat_khu_client():
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Chat KHU 호출에는 python-dotenv 패키지가 필요합니다.") from exc

    load_dotenv(BASE_DIR / ".env")
    load_dotenv()
    api_key = os.getenv("CHAT_KHU")
    if not api_key:
        raise RuntimeError("CHAT_KHU 환경변수를 찾을 수 없습니다. DATA_JOUR_BASE_DIR의 .env를 확인하세요.")
    print("Chat KHU API Key 로드 성공")
    return api_key


def call_chat_khu(api_key: str, model: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    response = requests.post(
        CHAT_KHU_CHAT_COMPLETIONS_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} {response.text}")

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Chat KHU 응답 형식이 예상과 다릅니다: {data}") from exc


def load_success_post_ids(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("post_id") and row.get("_status") == "success":
                completed.add(row["post_id"])
    return completed


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    output_df = pd.DataFrame(rows)
    if not output_df.empty and "post_id" in output_df.columns:
        output_df = output_df.drop_duplicates(subset=["post_id"], keep="last")
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def apply_resume_and_limits(selected: pd.DataFrame, config: AnalysisConfig, output_jsonl: Path) -> pd.DataFrame:
    if config.resume:
        completed = load_success_post_ids(output_jsonl)
        if completed:
            before = len(selected)
            selected = selected[~selected["post_id"].isin(completed)].reset_index(drop=True)
            print(f"이어하기 모드: 기존 성공 분석 {before - len(selected):,}건 제외")
        else:
            print("이어하기 모드: 기존 성공 분석 결과가 없어 처음부터 시작합니다.")
    else:
        print("--no-resume 지정: 기존 결과를 무시하고 다시 분석합니다.")

    if config.sample_per_category:
        selected = (
            selected.groupby("core_keyword", group_keys=False)
            .head(config.sample_per_category)
            .reset_index(drop=True)
        )
    if config.limit:
        selected = selected.head(config.limit).reset_index(drop=True)
    return selected


def is_limit_or_credential_error(exc: Exception) -> bool:
    message = str(exc).lower()
    patterns = [
        "quota",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "429",
        "402",
        "6005",
        "6008",
        "insufficient",
        "credit",
        "크레딧",
        "한도",
        "부족",
        "billing",
        "payment",
        "exceeded",
        "exhausted",
        "too many",
        "reservation",
        "permission",
        "unauthorized",
        "unauthenticated",
        "api key",
        "apikey",
        "403",
        "401",
    ]
    return any(pattern in message for pattern in patterns)


def run_chat_khu_analysis(df: pd.DataFrame, config: AnalysisConfig) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    prompt_template = load_prompt_template(config.prompt_file)
    selected = select_llm_input(df, config, apply_limits=False)
    output_stem = get_output_stem(config)

    output_jsonl = OUTPUT_DIR / f"{output_stem}_chat_khu_post_analysis.jsonl"
    output_csv = OUTPUT_DIR / f"{output_stem}_chat_khu_post_analysis.csv"

    selected = apply_resume_and_limits(selected, config, output_jsonl)

    print(f"Chat KHU 분석 대상: {len(selected):,}건")
    print(f"사용 모델: {config.model}")
    print(f"결과 JSONL: {output_jsonl}")

    api_key = get_chat_khu_client()
    consecutive_errors = 0
    stop_reason = ""

    try:
        for idx, row in selected.iterrows():
            post_id = str(row.get("post_id", ""))
            text = build_llm_text(row, config.max_text_chars)
            prompt = prompt_template.format(text=text)
            result = {
                "post_id": post_id,
                "core_keyword": row.get("core_keyword", ""),
                "source": row.get("source", ""),
                "link": row.get("link", ""),
                "title": row.get("title", ""),
                "source_file": row.get("source_file", ""),
                "comments_file_key": row.get("comments_file_key", ""),
                "model": config.model,
                "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            print(f"[{idx + 1}/{len(selected)}] {post_id}")
            raw_response = ""
            should_stop = False
            try:
                raw_response = call_chat_khu(api_key, config.model, prompt)
                parsed = parse_json_response(raw_response)
                result.update(parsed)
                result["_status"] = "success"
                result["_raw"] = ""
                consecutive_errors = 0
            except json.JSONDecodeError as exc:
                result["_status"] = "json_error"
                result["_error"] = str(exc)
                result["_raw"] = raw_response
                consecutive_errors += 1
            except Exception as exc:
                result["_status"] = "limit_or_auth_error" if is_limit_or_credential_error(exc) else "error"
                result["_error"] = str(exc)
                result["_raw"] = ""
                consecutive_errors += 1
                if is_limit_or_credential_error(exc):
                    should_stop = True
                    stop_reason = f"API 한도/크레딧/권한 오류 감지: {exc}"

            if should_stop:
                print(CREDIT_EXHAUSTION_MESSAGE)
                print(f"중단: {stop_reason}")
                break

            append_jsonl(output_jsonl, result)

            if consecutive_errors >= config.max_consecutive_errors:
                stop_reason = f"연속 오류 {consecutive_errors}건 발생"
                print(f"중단: {stop_reason}")
                break

            time.sleep(config.sleep_sec)
    except KeyboardInterrupt:
        stop_reason = "사용자 중단"
        print("\n사용자 중단 감지: 지금까지 저장된 JSONL을 CSV로 변환합니다.")
    finally:
        jsonl_to_csv(output_jsonl, output_csv)

    print(f"Chat KHU 분석 CSV 저장 완료: {output_csv}")
    if stop_reason:
        print(f"중단 사유: {stop_reason}")
    return output_csv


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="Analyze crawl_data_{category}.csv files.")
    parser.add_argument("--mode", choices=["audit", "export-llm-input", "llm"], default="audit")
    parser.add_argument("--input-csv", type=Path, default=None, help="분석할 단일 crawl_data_*.csv 파일 경로")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="--input-csv를 쓰지 않을 때만 사용할 주제 목록. 미지정 시 audit에서만 전체 주제를 점검합니다.",
    )
    parser.add_argument("--limit", type=int, default=None, help="LLM 입력 최대 건수")
    parser.add_argument("--sample-per-category", type=int, default=None, help="주제별 LLM 입력 최대 건수")
    parser.add_argument("--max-text-chars", type=int, default=12000, help="LLM 입력에 포함할 본문 최대 글자 수")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Chat KHU에 보낼 프롬프트 템플릿 파일. 생략하면 내장 시스템 프롬프트를 사용합니다.",
    )
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_LLM_SLEEP_SEC, help="API 호출 간 대기 시간")
    parser.add_argument("--max-consecutive-errors", type=int, default=5, help="연속 오류가 이 횟수에 도달하면 중단")
    parser.add_argument("--no-resume", action="store_true", help="기존 JSONL 완료 건을 무시하고 다시 호출")
    args = parser.parse_args()

    if args.mode in {"llm", "export-llm-input"} and args.input_csv is None:
        raise ValueError(
            f"{args.mode} 모드는 담당자별 단일 파일 분석을 위해 --input-csv를 반드시 지정해야 합니다."
        )

    categories = args.categories if args.categories is not None else CATEGORIES
    invalid = [category for category in categories if category not in CATEGORIES]
    if invalid:
        raise ValueError(f"지원하지 않는 주제입니다: {invalid}. 가능 주제: {CATEGORIES}")

    return AnalysisConfig(
        input_csv=args.input_csv,
        categories=categories,
        mode=args.mode,
        limit=args.limit,
        sample_per_category=args.sample_per_category,
        max_text_chars=args.max_text_chars,
        prompt_file=args.prompt_file,
        model=CHAT_KHU_MODEL,
        sleep_sec=args.sleep_sec,
        resume=not args.no_resume,
        max_consecutive_errors=args.max_consecutive_errors,
    )


def main() -> None:
    config = parse_args()
    posts = load_posts(config)
    posts = add_custom_indicators(posts)

    if config.mode == "audit":
        save_audit_outputs(posts, config)
    elif config.mode == "export-llm-input":
        save_audit_outputs(posts, config)
        export_llm_input(posts, config)
    elif config.mode == "llm":
        save_audit_outputs(posts, config)
        run_chat_khu_analysis(posts, config)


if __name__ == "__main__":
    main()
