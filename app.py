"""
n数試算ツール
クライアントの調査条件から回収見込み数・難易度を自動推計します
"""

import re
import streamlit as st
from google import genai
from google.genai import types
import json
import os
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# 定数・初期設定
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="n数試算ツール",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR        = "data"
PANEL_FILE      = os.path.join(DATA_DIR, "panel_data.json")
HISTORY_FILE    = os.path.join(DATA_DIR, "history.json")
CONFIG_FILE     = os.path.join(DATA_DIR, "config.json")
QA_HISTORY_FILE = "qa_history.json"
MODEL           = "gemini-2.0-flash"

AGE_GROUPS   = ["10代", "20代", "30代", "40代", "50代", "60代", "70代以上"]
GENDERS      = ["男性", "女性"]
UNKNOWN_KEYS = {"未取得", "わからない", "不明", "無回答"}

os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# カスタムCSS
# ─────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    /* ── フォント ── */
    html, body, [class*="css"] {
        font-family: 'Hiragino Sans', 'Yu Gothic UI', 'Noto Sans JP', sans-serif;
    }

    /* ── サイドバー背景 ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D2137 0%, #122B45 60%, #0A1A2B 100%);
        border-right: 1px solid #1E3A52;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption {
        color: #9BB8CF !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    /* サイドバー内の区切り線 */
    [data-testid="stSidebar"] hr {
        border-color: #1E3A52 !important;
        margin: 8px 0;
    }

    /* ── サイドバー ロゴボタン ── */
    [data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"]:first-of-type {
        background: transparent !important;
        border: none !important;
        color: #FFFFFF !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 8px 4px !important;
        text-align: left !important;
        letter-spacing: 0.02em;
    }

    /* ── サイドバー ナビゲーションボタン ── */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        border-radius: 8px;
        font-size: 13px;
        padding: 10px 14px;
        transition: background 0.15s, color 0.15s;
        border: none;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent;
        color: #9BB8CF;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.07);
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #0078C8;
        color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0,120,200,0.35);
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: #005FA0;
    }

    /* ── 「新しい調査」ボタン ── */
    [data-testid="stSidebar"] .reset-btn > button {
        background: rgba(255, 97, 42, 0.15) !important;
        color: #FF8C5A !important;
        border: 1px solid rgba(255, 97, 42, 0.3) !important;
    }
    [data-testid="stSidebar"] .reset-btn > button:hover {
        background: rgba(255, 97, 42, 0.28) !important;
    }

    /* ── メインエリア背景 ── */
    .main .block-container {
        background-color: #F4F7FB;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    /* ── ページタイトル ── */
    h1 {
        color: #0D2137;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    h2, h3 {
        color: #1A3A58;
    }

    /* ── メトリクスカード ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D8E6F0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 2px 10px rgba(0,30,60,0.07);
    }
    [data-testid="stMetricLabel"] {
        color: #4A6A8A !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0078C8 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #2EA87A !important;
        font-weight: 600 !important;
    }

    /* ── プライマリボタン（メインエリア） ── */
    .main .stButton > button[kind="primary"] {
        background: #0078C8;
        border-color: #0078C8;
        border-radius: 8px;
        font-weight: 700;
        padding: 10px 24px;
        font-size: 15px;
        box-shadow: 0 2px 10px rgba(0,120,200,0.30);
        transition: background 0.15s;
    }
    .main .stButton > button[kind="primary"]:hover {
        background: #005FA0;
        border-color: #005FA0;
    }

    /* ── セカンダリボタン（メインエリア） ── */
    .main .stButton > button[kind="secondary"] {
        border-radius: 8px;
        border-color: #C2D8EC;
        color: #2A5070;
        transition: background 0.15s;
    }
    .main .stButton > button[kind="secondary"]:hover {
        background: #E4F0FA;
    }

    /* ── ボーダー付きコンテナ ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border-color: #D8E6F0 !important;
        background: #FFFFFF !important;
        padding: 4px 4px !important;
    }

    /* ── アラート系 ── */
    .stAlert {
        border-radius: 10px;
    }
    [data-testid="stAlert"][kind="info"] {
        background: #EAF4FB;
        border-left-color: #0078C8;
    }
    [data-testid="stAlert"][kind="success"] {
        background: #E6F7F0;
        border-left-color: #2EA87A;
    }
    [data-testid="stAlert"][kind="warning"] {
        background: #FFF5E6;
        border-left-color: #F59E0B;
    }

    /* ── テキスト入力 ── */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border-color: #C2D8EC;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0078C8;
        box-shadow: 0 0 0 2px rgba(0,120,200,0.15);
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #D8E6F0;
        border-radius: 10px;
    }
    [data-testid="stExpanderToggleIcon"] {
        color: #0078C8;
    }

    /* ── テーブル ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #D8E6F0;
    }

    /* ── 区切り線 ── */
    hr {
        border-color: #D8E6F0;
    }

    /* ── サイドバー パネル情報 ── */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        border-radius: 8px;
        font-size: 12px;
    }

    /* ── サイドバー input ── */
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08);
        border-color: #2A4A66;
        color: #FFFFFF;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input::placeholder {
        color: #5A7A96;
    }

    /* ── サイドバー Expander ── */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,0.05);
        border-color: #1E3A52;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
        color: #9BB8CF;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# データ管理
# ─────────────────────────────────────────────────────────────────

def _load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_panel() -> dict:
    return _load_json(PANEL_FILE, {
        "total": 0,
        "age":   {a: 0 for a in AGE_GROUPS},
        "gender": {"男性": 0, "女性": 0},
        "prefecture": {},
        "attributes": {},
    })


def load_config() -> dict:
    return _load_json(CONFIG_FILE, {"api_key": ""})


def save_config(cfg: dict):
    _save_json(CONFIG_FILE, cfg)


def append_history(entry: dict):
    """計算履歴をセッション（ブラウザ単位）に保存する。他のユーザーとは共有されない。"""
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    st.session_state.session_history.insert(0, entry)
    if len(st.session_state.session_history) > 100:
        st.session_state.session_history = st.session_state.session_history[:100]


def load_history() -> list:
    return st.session_state.get("session_history", [])


def load_qa_history() -> list:
    return _load_json(QA_HISTORY_FILE, [])


def search_similar_cases(condition: str, qa_history: list, top_k: int = 5) -> list:
    if not qa_history or not condition.strip():
        return []

    def bigrams(text: str) -> set:
        t = text.replace(" ", "").replace("　", "")
        return {t[i:i+2] for i in range(len(t) - 1)} if len(t) >= 2 else set()

    q_bi = bigrams(condition)
    if not q_bi:
        return []

    scored = []
    for rec in qa_history:
        haystack = " ".join([
            rec.get("target", ""),
            rec.get("industry", ""),
            rec.get("conditions", ""),
            rec.get("category", ""),
        ])
        h_bi = bigrams(haystack)
        overlap = len(q_bi & h_bi) / len(q_bi) if q_bi else 0
        if overlap > 0.08:
            scored.append((overlap, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


def format_similar_cases(cases: list) -> str:
    if not cases:
        return ""
    lines = ["【過去の類似回収実績（参考）】",
             "以下は同類の条件で実際に試算した回収実績です。behavioral_rateの推計に活用してください。"]
    for c in cases:
        n_str = c.get("n_raw", "不明")
        target = c.get("target", "")
        industry = c.get("industry", "")
        notes = c.get("notes", "").replace("\n", " ")[:80]
        conditions = c.get("conditions", "")
        line = f"・対象：{target}"
        if industry and industry != "指定なし":
            line += f"（{industry}）"
        line += f" → 回収実績：{n_str}人"
        if conditions:
            line += f"　条件：{conditions[:40]}"
        if notes:
            line += f"　備考：{notes}"
        lines.append(line)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# モニタスCSV パーサー
# ─────────────────────────────────────────────────────────────────

def _parse_monitas_csv(df: pd.DataFrame) -> dict:
    def to_int(val) -> int:
        if pd.isna(val):
            return 0
        s = str(val).replace(",", "").strip()
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return 0

    gender: dict     = {"男性": 0, "女性": 0}
    age_raw: dict    = {}
    prefecture: dict = {}
    raw_attrs: dict  = {}
    current_category: str | None = None

    for _, row in df.iterrows():
        cells = [str(v).strip() if not pd.isna(v) else "" for v in row]
        while len(cells) < 7:
            cells.append("")
        col0, col1, col2 = cells[0], cells[1], cells[2]

        if not col0 and not col1:
            continue
        if col0:
            current_category = col0
        if not current_category or not col1:
            continue

        count = to_int(col2)

        if current_category == "性別":
            if col1 in ("男性", "女性"):
                gender[col1] = count
        elif current_category == "年代":
            age_raw[col1] = count
        elif current_category == "都道府県":
            prefecture[col1] = count
        else:
            if current_category not in raw_attrs:
                raw_attrs[current_category] = {}
            raw_attrs[current_category][col1] = count

    age = {a: 0 for a in AGE_GROUPS}
    age_70plus = 0
    for age_name, cnt in age_raw.items():
        if age_name in ("70代", "80代", "90代以上"):
            age_70plus += cnt
        elif age_name in AGE_GROUPS:
            age[age_name] = cnt
    age["70代以上"] = age_70plus

    total = gender["男性"] + gender["女性"]
    if total == 0:
        total = sum(age.values())

    attributes: dict = {}
    for cat_name, cat_data in raw_attrs.items():
        answered = sum(v for k, v in cat_data.items() if k not in UNKNOWN_KEYS)
        coverage = round(min(answered / total, 1.0), 3) if total > 0 else 0.0
        attributes[cat_name] = {"data": cat_data, "coverage": coverage}

    return {
        "total": total, "age": age, "gender": gender,
        "prefecture": prefecture, "attributes": attributes,
    }


# ─────────────────────────────────────────────────────────────────
# 難易度判定
# ─────────────────────────────────────────────────────────────────

def _difficulty(adj_inc: float, adj_est_min: int) -> tuple[str, str, str]:
    if adj_est_min >= 1000:
        return "易", "success", "回収しやすい条件です。十分な人数が見込めます。"
    elif adj_est_min >= 500:
        return "普通", "info", "標準的な回収難易度です。"
    elif adj_est_min >= 100:
        return "難", "warning", "回収に工夫が必要な条件です。緩和措置も検討してください。"
    else:
        return "困難", "error", "回収が非常に困難です。条件の見直しを強く推奨します。"


# ─────────────────────────────────────────────────────────────────
# Claude API：条件分析
# ─────────────────────────────────────────────────────────────────

def analyze_condition(api_key: str, condition: str, panel: dict,
                      has_multi_group: bool = False,
                      similar_cases: list | None = None) -> dict:

    client = genai.Client(api_key=api_key)

    lines = [f"パネル総数: {panel['total']:,}人", "", "【年代別】（完全回収）"]
    for age, n in panel["age"].items():
        lines.append(f"  {age}: {n:,}人")
    lines += ["", "【性別】（完全回収）"]
    for g, n in panel["gender"].items():
        lines.append(f"  {g}: {n:,}人")
    if panel.get("prefecture"):
        top = sorted(panel["prefecture"].items(), key=lambda x: x[1], reverse=True)[:10]
        lines += ["", "【都道府県別（上位10件）】（完全回収）"]
        for p, n in top:
            lines.append(f"  {p}: {n:,}人")
    if panel.get("attributes"):
        for cat_name, cat_info in panel["attributes"].items():
            cov = cat_info.get("coverage", 0)
            reliability = "完全回収" if cov >= 0.85 else f"部分回収（回答率{cov*100:.0f}%）"
            data = cat_info.get("data", {})
            lines.append(f"\n【{cat_name}】（{reliability}）")
            items = sorted(data.items(), key=lambda x: x[1], reverse=True)
            for val, cnt in items[:15]:
                lines.append(f"  {val}: {cnt:,}人")
            if len(items) > 15:
                lines.append(f"  ※他{len(items)-15}件省略")
    panel_text = "\n".join(lines)

    import re as _re
    has_or  = has_multi_group or bool(_re.search(r'\bOR\b', condition))
    or_hint = (
        "\n⚠️ この入力には複数の対象グループ（OR条件）が含まれています。"
        "必ず is_multi_group=true にして target_groups を使ってください。"
    ) if has_or else ""

    similar_text = format_similar_cases(similar_cases or [])

    user_prompt = f"""以下の入力から調査したいターゲット条件を読み取り、モニターパネルからの回収見込みを推計してください。{or_hint}

入力は箇条書き・口語・質問・相談文のどれでも構いません。
「〇〇って取れる？」「こういう条件どう？」でも正確に解釈してください。
複数の解釈がある場合は調査実務として最も現実的な解釈を選んでください。

【入力】
{condition}

【モニターパネルデータ】
{panel_text}
{similar_text}

次のJSON形式のみで回答してください（コードブロック不要・JSONオブジェクトのみ）：

{{
  "condition_summary": "条件を1〜2行で要約",
  "is_multi_group": false,
  "include_ages": ["30代", "40代", "50代"],
  "exclude_ages": ["10代", "70代以上"],
  "exclude_reason": "除外した理由",
  "gender_specified": false,
  "include_genders": ["男性", "女性"],
  "prefecture_specified": false,
  "include_prefectures": [],
  "attribute_filters": [
    {{"category": "カテゴリ名", "values": ["属性値1"], "is_reliable": true, "note": "理由"}}
  ],
  "target_groups": [
    {{
      "description": "グループの説明",
      "attribute_filters": [{{"category": "職種", "values": ["値1"], "is_reliable": false, "note": "理由"}}],
      "behavioral_rate": 0.40,
      "behavioral_rate_min": 0.28,
      "behavioral_rate_max": 0.52
    }}
  ],
  "behavioral_rate": 0.05,
  "behavioral_rate_min": 0.03,
  "behavioral_rate_max": 0.07,
  "behavioral_reasoning": "推計根拠",
  "confidence": "medium",
  "difficulty_reason": "難易度の理由",
  "relaxation_suggestions": [
    {{"action": "緩和内容", "additional_est": 1000, "trade_off": "デメリット", "recommended": true}}
  ],
  "warnings": []
}}

※ is_multi_group=false の場合は attribute_filters を使い target_groups は [] にする。
※ is_multi_group=true の場合は target_groups に各グループを入れ attribute_filters は [] にする。

【判断基準】※すべて「保守的・厳しめ」に設定すること

・is_multi_group:
  以下のいずれかに該当する場合は必ず true にする。
  ① 「OR」「または」「および」で異なる職種・職業・役割が並列している
  ② 同一人物では同時に成立しえない複数の属性が並列している
  ③ プロンプトに「⚠️ OR（複数グループ）の意味が含まれています」と書かれている
  → is_multi_group=true の場合、target_groups に各グループを分けて記載し attribute_filters は [] にする
  例）「人事担当者 OR 学校教職員」→ true（2グループ）
  例）「製造業の購買担当者」→ false（1種類）

・include_ages: 条件から合理的に推論して必ず設定すること。絶対に空リスト [] にしない。
  - 条件に年代が明示されていなくても、職業・状況・ライフステージから推論する
  - 「確実に対象外」の年代を除外し、残りをすべて含める
  例）NISA投資歴3年以内 → ["20代","30代","40代","50代","60代"]
  例）子育て中 → ["20代","30代","40代","50代"]
  例）前職・前々職が自衛官 → 入隊最短でも18歳、退職後に就業している世代を想定
    → ["30代","40代","50代","60代","70代以上"]（10代・20代は退職OBとして社会人である可能性が低い）
  例）現役の大学生 → ["10代","20代"]
  例）介護経験者 → ["30代","40代","50代","60代","70代以上"]

・gender_specified: 条件から男女比が偏ることが明らかな場合は true にして絞り込む
  - 明示されていなくても職業・状況から推論してよい
  例）自衛官・消防士・警察官 → 男性比率が非常に高い → gender_specified=true, include_genders=["男性"]
  例）保育士・看護師・産婦人科医 → 女性比率が高い → gender_specified=true, include_genders=["女性"]
  例）経営者 → 男性比率は高いが女性も一定数いる → gender_specified=false（両方含める）
  例）条件に「女性」「男性」と明示 → 必ず gender_specified=true

・attribute_filters（単一グループ時）: 条件に「直接かつ明確に」関連するカテゴリのみ。
  「間違いなく当てはまる」値のみ（周辺的な値は含めない）

  【重要：相関フィルタの二重使用禁止】
  以下の組み合わせは「二重フィルタ」になるため禁止。どちらか一方（より具体的な方）のみ使うこと。
  × 職種 + 職業   → 職種のみ使う（職種の人は職業の中にいる）
  × 職種 + 役職   → 職種のみ使う（担当者・責任者という役職条件は behavioral_rate で表現する）
  × 業種 + 職業   → 業種のみ使う
  × 役職 + 職業   → 役職のみ使う

  【重要：職場規模・企業規模は attribute_filters に絶対に含めない】
  「従業員100名以上」「中小企業」「大企業」などの企業規模・職場規模の条件は
  attribute_filters に含めず、必ず behavioral_rate で表現すること。
  × 職種 + 職場規模 → 職種のみ attribute_filters に使い、企業規模の絞り込みは behavioral_rate で表現する
  例）「100名以上企業の経理担当者」
    → attribute_filters = [{"category": "職種", "values": ["財務／会計／経理"]}]
    → behavioral_rate = 0.40〜0.55（経理職種の中で100名以上企業に勤務している割合）

  例）「経営企画担当者」→ 職種=経営企画/事業企画 のみ。役職フィルタ不要。
     behavioral_rate で「経営企画職種者の中での担当者割合（≒0.80）」を表現する。
  例）「製造業の部長クラス」→ 業種=製造業系 + 役職=部長クラス は OK（これらは独立）

  【重要：職種・役職の値は必ずパネルデータに表示された完全一致の文字列を使うこと】
  例）「経営企画」→ パネルデータの「経営企画／事業企画」を使う（「営業企画」は別物）
  例）「財務・経理」→ パネルデータの「財務／会計／経理」を使う
  値の一部分だけで判断せず、パネルデータに表示されている選択肢一覧から正確に選ぶこと。

・behavioral_rate: 「attribute_filters で絞った後の対象者のうち、さらに条件に合う割合」
  ※ attribute_filters で職種・業種を既に絞っている場合、behavioral_rate は
    「その職種の人の中で該当業務・行動をしている割合」として設定する。
    パネル全体に対する割合ではない。

  【残り条件のタイプ別目安】（attribute_filters 適用後の残条件に対して）

  A. 現在進行中の業務担当・関与（今まさにその業務を担当している）→ 0.35〜0.60
     例）人事職種の中でオンライン試験実施を担当        → 0.40〜0.55
     例）教職員の中でオンライン試験運営に携わる        → 0.45〜0.60
     例）IT企業社員の中でクラウド移行プロジェクト担当  → 0.25〜0.40
     例）経営企画・財務経理職種の担当者・責任者全般    → 0.45〜0.60

  B. 属性条件のみで行動条件なし                       → 0.55〜0.70
     ※ 業種フィルタは自己申告精度が低いため上限 0.50

  C. 保有・利用中（〇〇を持っている・使っている）     → 0.15〜0.30
  D. 過去の行動経験（〇年以内に〇〇した）             → 0.05〜0.20
  E. 意向・態度（関心がある・検討中）                 → 0.10〜0.25

  ※ 独立した行動条件が複数重なる場合のみ 60〜75% に下げる
    （attribute_filters で既に絞った職種条件はここに含めない）

  【数値例】
  - 人事職種 → オンライン試験実施担当       : 0.40〜0.55
  - 教職員職種 → オンライン試験運営担当     : 0.45〜0.60
  - NISA放置型投資歴3年以内               : 0.05〜0.09
  - IT業界勤務のエンジニア（業種フィルタ後）: 0.25〜0.35
  - 会社員（行動条件なし）                 : 0.55〜0.60
  - 製造業の購買担当部長クラス             : 0.04〜0.08

  behavioral_rate_min は behavioral_rate の 60〜70% に設定すること

・relaxation_suggestions: 3件。条件に書かれていない次元（年代・地域・職業など）も自由に提案してよい。
  - 3件のうち最も効果的な1件に recommended: true を設定すること
  - trade_offに必ずデメリットを記載する

・confidence: high（公的統計引用可）/ medium（業界推計あり）/ low（根拠が薄い）
・性別・地域が条件に明示されていなければ specified = false
"""

    res = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction="あなたは市場調査・パネル調査の専門家です。必ず有効なJSONのみを返してください。マークダウンのコードブロックは使わず、JSONオブジェクトをそのまま返してください。",
            max_output_tokens=4000,
        ),
    )
    text = res.text.strip()

    if "```" in text:
        for block in text.split("```")[1::2]:
            candidate = block.lstrip("json").strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group())

    raise json.JSONDecodeError("JSONを抽出できませんでした", text, 0)


# ─────────────────────────────────────────────────────────────────
# 計算ロジック
# ─────────────────────────────────────────────────────────────────

def _remove_redundant_filters(attr_filters: list) -> list:
    cats = {af.get("category", "") for af in attr_filters}
    remove = set()
    if "職種" in cats:
        remove |= {"職業", "役職"}
    elif "役職" in cats:
        remove |= {"職業"}
    if "業種" in cats:
        remove |= {"職業"}
    if remove:
        return [af for af in attr_filters if af.get("category") not in remove]
    return attr_filters


def _calc_attr_filters(panel: dict, attr_filters: list, total: int) -> tuple[float, list]:
    attr_filters = _remove_redundant_filters(attr_filters)
    attr_combined_ratio = 1.0
    details = []
    for af in attr_filters:
        cat_name = af.get("category", "")
        values   = af.get("values", [])
        if not cat_name or not values:
            continue
        cat_info = panel.get("attributes", {}).get(cat_name, {})
        cat_data = cat_info.get("data", {})
        coverage = cat_info.get("coverage", 1.0)
        if not cat_data:
            continue
        matched = sum(cat_data.get(v, 0) for v in values)
        ratio   = matched / total if total > 0 else 0.0
        attr_combined_ratio *= ratio
        details.append({
            "category": cat_name,
            "values":   values,
            "matched":  matched,
            "ratio":    round(ratio * 100, 2),
            "coverage": coverage,
            "reliable": af.get("is_reliable", coverage >= 0.85),
            "note":     af.get("note", ""),
        })
    return attr_combined_ratio, details


def _calc_one_group(panel: dict, group: dict, age_ratio: float,
                    gender_ratio: float, pref_ratio: float,
                    activity_rate: float) -> dict:
    total = panel["total"]
    attr_ratio, attr_details = _calc_attr_filters(
        panel, group.get("attribute_filters", []), total
    )
    demo_base = total * age_ratio * gender_ratio * pref_ratio * attr_ratio

    brate     = min(float(group.get("behavioral_rate",     1.0)), 1.0)
    brate_min = min(float(group.get("behavioral_rate_min", brate * 0.6)), 1.0)
    brate_max = min(float(group.get("behavioral_rate_max", brate * 1.4)), 1.0)

    est     = demo_base * brate
    est_min = demo_base * brate_min
    est_max = demo_base * brate_max

    return {
        "description":         group.get("description", ""),
        "demo_base":           int(demo_base),
        "attr_filter_details": attr_details,
        "behavioral_rate":     brate,
        "est":                 int(est),
        "est_min":             int(est_min),
        "est_max":             int(est_max),
        "adj_est":             int(est     * activity_rate),
        "adj_est_min":         int(est_min * activity_rate),
        "adj_est_max":         int(est_max * activity_rate),
    }


def calculate(panel: dict, analysis: dict, activity_rate: float = 0.45) -> dict | None:
    total = panel["total"]
    if total == 0:
        return None

    include_ages = analysis.get("include_ages", AGE_GROUPS) or AGE_GROUPS
    age_base  = sum(panel["age"].get(a, 0) for a in include_ages)
    age_ratio = age_base / total if total > 0 else 1.0

    gender_ratio = 1.0
    if analysis.get("gender_specified"):
        genders  = analysis.get("include_genders", GENDERS)
        g_count  = sum(panel["gender"].get(g, 0) for g in genders)
        gender_ratio = g_count / total if total > 0 else 1.0

    pref_ratio = 1.0
    if analysis.get("prefecture_specified") and analysis.get("include_prefectures"):
        prefs = analysis["include_prefectures"]
        if panel.get("prefecture"):
            p_count   = sum(panel["prefecture"].get(p, 0) for p in prefs)
            pref_ratio = p_count / total if total > 0 else 1.0

    target_groups_raw = analysis.get("target_groups", [])
    is_multi = analysis.get("is_multi_group", False) and len(target_groups_raw) > 0

    if is_multi:
        group_results = [
            _calc_one_group(panel, g, age_ratio, gender_ratio, pref_ratio, activity_rate)
            for g in target_groups_raw
        ]
        adj_est     = sum(g["adj_est"]     for g in group_results)
        adj_est_min = sum(g["adj_est_min"] for g in group_results)
        adj_est_max = sum(g["adj_est_max"] for g in group_results)
        est         = sum(g["est"]         for g in group_results)
        est_min     = sum(g["est_min"]     for g in group_results)
        est_max     = sum(g["est_max"]     for g in group_results)
        demo_base   = sum(g["demo_base"]   for g in group_results)
        attr_filter_details = []
        brate = sum(g["behavioral_rate"] for g in group_results) / len(group_results)
    else:
        attr_combined_ratio, attr_filter_details = _calc_attr_filters(
            panel, analysis.get("attribute_filters", []), total
        )
        demo_base = total * age_ratio * gender_ratio * pref_ratio * attr_combined_ratio

        brate     = min(float(analysis.get("behavioral_rate",     1.0)), 1.0)
        brate_min = min(float(analysis.get("behavioral_rate_min", brate * 0.6)), 1.0)
        brate_max = min(float(analysis.get("behavioral_rate_max", brate * 1.4)), 1.0)

        est     = demo_base * brate
        est_min = demo_base * brate_min
        est_max = demo_base * brate_max

        adj_est     = est     * activity_rate
        adj_est_min = est_min * activity_rate
        adj_est_max = est_max * activity_rate
        group_results = []

    est     = max(est,     0)
    est_min = max(est_min, 0)
    est_max = max(est_max, 0)
    adj_est     = max(adj_est,     0)
    adj_est_min = max(adj_est_min, 0)
    adj_est_max = max(adj_est_max, 0)

    adj_inc     = adj_est     / total * 100
    adj_inc_min = adj_est_min / total * 100
    adj_inc_max = adj_est_max / total * 100

    difficulty, diff_color, diff_desc = _difficulty(adj_inc, int(adj_est_min))

    return {
        "total":               total,
        "age_base":            int(age_base),
        "demo_base":           int(demo_base),
        "gender_ratio":        gender_ratio,
        "pref_ratio":          pref_ratio,
        "is_multi_group":      is_multi,
        "group_results":       group_results,
        "attr_filter_details": attr_filter_details,
        "include_ages":        include_ages,
        "exclude_ages":        analysis.get("exclude_ages", []),
        "behavioral_rate":     brate,
        "est":                 int(est),
        "est_min":             int(est_min),
        "est_max":             int(est_max),
        "activity_rate":       activity_rate,
        "adj_est":             int(adj_est),
        "adj_est_min":         int(adj_est_min),
        "adj_est_max":         int(adj_est_max),
        "adj_inc":             round(adj_inc,     2),
        "adj_inc_min":         round(adj_inc_min, 2),
        "adj_inc_max":         round(adj_inc_max, 2),
        "difficulty":          difficulty,
        "diff_color":          diff_color,
        "diff_desc":           diff_desc,
    }


# ─────────────────────────────────────────────────────────────────
# レポート生成
# ─────────────────────────────────────────────────────────────────

def make_report(condition: str, analysis: dict, r: dict) -> str:
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    sep = "━" * 42
    lines = [
        sep,
        "  市場調査 回収見込み推計レポート",
        f"  作成日時：{now}",
        sep,
        "",
        "■ 調査ターゲット条件",
        f"  {condition}",
        "",
        "■ 条件の解釈",
        f"  {analysis.get('condition_summary', '')}",
        "",
        f"■ 回収難易度：{r['difficulty']}",
        f"  {analysis.get('difficulty_reason', '')}",
        "",
        "■ 回収見込み（パネル稼働率補正後）",
        f"  一般的な見込み ：約 {r['adj_est']:,}人（出現率 {r['adj_inc']}%）",
        f"  固めの見込み   ：約 {r['adj_est_min']:,}人（出現率 {r['adj_inc_min']}%）",
        f"  ※パネル稼働率補正45%適用（パネル実回答率×スクリーニング完了率）",
        "",
    ]

    if analysis.get("relaxation_suggestions"):
        lines.append("■ n数を増やしたい場合の緩和措置")
        for s in analysis["relaxation_suggestions"]:
            additional = s.get("additional_est", 0)
            lines.append(f"  ・{s.get('action', '')}")
            lines.append(f"    → 追加で約 {additional:,}人の回収見込み（AI推計）")
            lines.append(f"    └ トレードオフ：{s.get('trade_off', '')}")
        lines.append("")

    lines += [
        "■ 推計の根拠",
        f"  対象年代：{' / '.join(r['include_ages'])}",
        f"  除外年代：{' / '.join(r['exclude_ages'])}",
        f"    └ 理由：{analysis.get('exclude_reason', '')}",
    ]

    if r.get("attr_filter_details"):
        lines.append("  属性フィルタ：")
        for af in r["attr_filter_details"]:
            tag = "完全回収" if af.get("reliable") else f"部分回収（{af['coverage']*100:.0f}%）"
            lines.append(
                f"    ・{af['category']}＝{' / '.join(af['values'])}"
                f"　{af['matched']:,}人（{af['ratio']}%）[{tag}]"
            )

    lines += [
        f"  行動・態度条件の出現率：約 {r['behavioral_rate']*100:.1f}%",
        f"  {analysis.get('behavioral_reasoning', '')}",
        "",
        "※ 本推計はAIによる統計的推定値です。",
        "  実際の出現率はスクリーニング調査での確認を推奨します。",
        sep,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 結果表示
# ─────────────────────────────────────────────────────────────────

def reset_calc():
    for key in ["calc_condition", "calc_analysis", "calc_results"]:
        st.session_state.pop(key, None)
    st.session_state.n_filters = 1
    st.session_state.n_groups  = 1
    for i in range(5):
        st.session_state.pop(f"f_{i}", None)
        st.session_state.pop(f"g_{i}", None)


def show_results(condition: str, analysis: dict, r: dict):

    # ── 難易度バッジ ─────────────────────────────────────────────
    color_map = {
        "success": ("🟢", "#28a745"),
        "info":    ("🔵", "#17a2b8"),
        "warning": ("🟠", "#fd7e14"),
        "error":   ("🔴", "#dc3545"),
    }
    icon, _ = color_map.get(r["diff_color"], ("⚪", "#6c757d"))

    st.subheader(f"{icon} 回収難易度：{r['difficulty']}")
    st.caption(r["diff_desc"])

    # ── 条件の解釈 ────────────────────────────────────────────────
    st.info(f"**条件の解釈：** {analysis.get('condition_summary', '')}")

    # ── 回収見込み 2段表示 ───────────────────────────────────────
    st.divider()
    st.write(f"**📊 回収見込み**　（回収率補正 {r['activity_rate']*100:.0f}%）")

    col_gen, col_firm = st.columns(2)
    with col_gen:
        st.metric(
            "一般的な見込み",
            f"{r['adj_est']:,}人",
            f"出現率 {r['adj_inc']}%",
        )
        st.caption("行動・態度条件の中央値で推計")
    with col_firm:
        st.metric(
            "固めの見込み",
            f"{r['adj_est_min']:,}人",
            f"出現率 {r['adj_inc_min']}%",
        )
        st.caption("行動・態度条件の下限値で推計（保守的）")

    if r.get("is_multi_group") and r.get("group_results"):
        st.caption("▼ ターゲットグループ別の内訳（上記は合算値）")
        cols_grp = st.columns(len(r["group_results"]))
        for i, g in enumerate(r["group_results"]):
            with cols_grp[i]:
                st.write(f"**{g['description']}**")
                st.write(f"一般的：{g['adj_est']:,}人")
                st.write(f"固め：{g['adj_est_min']:,}人")
                if g.get("attr_filter_details"):
                    for af in g["attr_filter_details"]:
                        tag = "🟢" if af.get("reliable") else "🟡"
                        st.caption(f"{tag} {af['category']}：{af['matched']:,}人")

    # ── 緩和措置アドバイス ───────────────────────────────────────
    suggestions = analysis.get("relaxation_suggestions", [])
    if suggestions:
        st.divider()
        st.write("**💡 n数を増やしたい場合の緩和措置**")
        st.caption("条件を変えた場合の追加回収見込み（AI推計・概算値）")
        sorted_suggestions = sorted(suggestions, key=lambda s: not s.get("recommended", False))
        for s in sorted_suggestions:
            action         = s.get("action", "")
            additional     = s.get("additional_est", 0)
            trade_off      = s.get("trade_off", "")
            is_recommended = s.get("recommended", False)
            with st.container(border=True):
                if is_recommended:
                    st.markdown("⭐ **おすすめ**")
                st.write(f"**{action}**")
                st.write(f"追加で約 **{additional:,}人** の回収見込み")
                if trade_off:
                    st.caption(f"⚠️ トレードオフ：{trade_off}")

    # ── 根拠・詳細（折りたたみ） ─────────────────────────────────
    st.divider()
    with st.expander("📖 推計根拠・計算の詳細を見る"):
        st.write("**対象・除外年代**")
        c1, c2 = st.columns(2)
        with c1:
            st.write("✅ 対象：" + " / ".join(r["include_ages"]))
        with c2:
            if r["exclude_ages"]:
                st.write("❌ 除外：" + " / ".join(r["exclude_ages"]))
                st.caption(analysis.get("exclude_reason", ""))

        if r.get("attr_filter_details"):
            st.write("**属性フィルタ**")
            for af in r["attr_filter_details"]:
                tag = "🟢 完全回収" if af.get("reliable") else f"🟡 部分回収（{af['coverage']*100:.0f}%）"
                st.write(
                    f"・**{af['category']}**：{' / '.join(af['values'])}"
                    f"　→ {af['matched']:,}人（{af['ratio']}%）　{tag}"
                )
                if not af.get("reliable"):
                    st.caption("　⚠️ 未回答者が多いため、実際の人数はこれより多い可能性があります")

        st.write("**推計根拠**")
        st.write(analysis.get("behavioral_reasoning", ""))

        conf_map = {
            "high":   "🟢 高（公的統計データあり）",
            "medium": "🟡 中（業界推計・調査事例あり）",
            "low":    "🔴 低（不確実性が高い）",
        }
        st.write(f"推計信頼度：{conf_map.get(analysis.get('confidence', 'medium'), '')}")

        if analysis.get("difficulty_reason"):
            st.write(f"難易度の理由：{analysis['difficulty_reason']}")

        if analysis.get("warnings"):
            st.warning("\n".join(f"• {w}" for w in analysis["warnings"]))

        st.write("**計算の内訳**")
        st.write(f"・パネル総数：{r['total']:,}人")
        st.write(f"・年代ベース：{r['age_base']:,}人")
        if r["gender_ratio"] < 1.0:
            st.write(f"・性別調整（×{r['gender_ratio']:.2f}）")
        if r["pref_ratio"] < 1.0:
            st.write(f"・地域調整（×{r['pref_ratio']:.2f}）")
        for af in r.get("attr_filter_details", []):
            st.write(f"・{af['category']}フィルタ（×{af['ratio']/100:.4f}）")
        st.write(f"・調整後母数：{r['demo_base']:,}人")
        st.write(f"・行動・態度条件の出現率：{r['behavioral_rate']*100:.1f}%")
        st.write(f"・理論推定人数：{r['est']:,}人")
        st.write(f"・回収率補正（×{r['activity_rate']:.2f}）→ {r['adj_est']:,}人")

    # ── 参照した過去事例 ─────────────────────────────────────────
    qa_hist = load_qa_history()
    similar = search_similar_cases(condition, qa_hist, top_k=5)
    if similar:
        with st.expander("📂 参照した過去の類似事例"):
            st.caption("推計に使用した過去の回収実績データ（類似度順）")
            rows = []
            for c in similar:
                rows.append({
                    "対象者": c.get("target", "")[:40],
                    "業界": c.get("industry", ""),
                    "回収実績": c.get("n_raw", ""),
                    "付加条件": c.get("conditions", "")[:30],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── レポート ─────────────────────────────────────────────────
    st.divider()
    report_text = make_report(condition, analysis, r)
    st.text_area(
        "📋 クライアント共有用レポート",
        report_text,
        height=300,
    )


# ─────────────────────────────────────────────────────────────────
# ページ①：出現率計算
# ─────────────────────────────────────────────────────────────────

def page_calculation(api_key: str, panel: dict):
    st.title("📊 回収見込み計算")

    if panel["total"] == 0:
        st.warning("⚠️ パネルデータが未設定です。管理者設定からパネルデータを更新してください。")
        return

    if "n_filters" not in st.session_state:
        st.session_state.n_filters = 1
    if "n_groups" not in st.session_state:
        st.session_state.n_groups = 1

    # ── ① 絞り込み条件 ──────────────────────────────────────────
    st.write("**① 絞り込み条件**")
    st.caption("対象者が**全員満たす**条件を入力してください（会社規模・年代・地域・雇用形態など）")

    ph_f = ["例）100名以上の企業", "例）30〜50代", "例）正社員", "例）関東在住", "例）男性"]
    filter_texts = []
    for i in range(st.session_state.n_filters):
        txt = st.text_input(
            f"絞り込み条件 {i+1}",
            key=f"f_{i}",
            placeholder=ph_f[i % len(ph_f)],
        )
        filter_texts.append(txt)

    ca, cb = st.columns(2)
    with ca:
        if st.button("＋ 絞り込み条件を追加", use_container_width=True, key="add_f"):
            if st.session_state.n_filters < 5:
                st.session_state.n_filters += 1
                st.rerun()
    with cb:
        if st.session_state.n_filters > 1:
            if st.button("－ 最後の条件を削除", use_container_width=True, key="del_f"):
                st.session_state.n_filters -= 1
                st.rerun()

    st.divider()

    # ── ② 対象グループ ──────────────────────────────────────────
    st.write("**② 対象グループ**")
    st.caption(
        "職種・部門・業種など、**どれかひとつに当てはまれば対象**になる条件を入力してください。"
        "　（複数入れると人数が合算されます）"
    )

    ph_g = ["例）経営担当者", "例）人事担当者", "例）現場責任者", "例）建設業界", "例）不動産業界"]
    group_texts = []
    for i in range(st.session_state.n_groups):
        txt = st.text_input(
            f"対象グループ {i+1}",
            key=f"g_{i}",
            placeholder=ph_g[i % len(ph_g)],
        )
        group_texts.append(txt)

    cc, cd = st.columns(2)
    with cc:
        if st.button("＋ 対象グループを追加", use_container_width=True, key="add_g"):
            if st.session_state.n_groups < 5:
                st.session_state.n_groups += 1
                st.rerun()
    with cd:
        if st.session_state.n_groups > 1:
            if st.button("－ 最後のグループを削除", use_container_width=True, key="del_g"):
                st.session_state.n_groups -= 1
                st.rerun()

    # ── 条件文字列の組み立て ──────────────────────────────────────
    f_parts = [t.strip() for t in filter_texts if t.strip()]
    g_parts = [t.strip() for t in group_texts  if t.strip()]
    has_multi_group = len(g_parts) >= 2

    lines = []
    if f_parts:
        lines.append("【絞り込み条件（AND：全員が満たす）】\n  " + "\n  ".join(f_parts))
    if g_parts:
        lines.append("【対象グループ（OR：どれかひとつでOK）】\n  " + "\n  ".join(g_parts))
    condition = "\n".join(lines)

    if condition.strip():
        with st.expander("📋 Claudeへの送信内容プレビュー"):
            st.code(condition, language=None)

    st.divider()
    activity_rate = 0.45

    if not api_key:
        st.warning("サイドバーでClaude API Keyを設定してください。")

    if st.button(
        "🔍 回収見込みを計算する",
        type="primary",
        disabled=(not condition.strip() or not api_key),
    ):
        with st.spinner("Claudeが分析中... （10〜20秒かかります）"):
            try:
                qa_hist = load_qa_history()
                similar = search_similar_cases(condition.strip(), qa_hist, top_k=5)
                analysis = analyze_condition(
                    api_key, condition.strip(), panel, has_multi_group, similar
                )
                results = calculate(panel, analysis, activity_rate)
            except json.JSONDecodeError:
                st.error("AIの応答を解析できませんでした。もう一度お試しください。")
                return
            except Exception as e:
                if "API_KEY_INVALID" in str(e) or "invalid" in str(e).lower() or "INVALID_ARGUMENT" in str(e):
                    st.error("APIキーが無効です。サイドバーで正しいGemini APIキーを入力してください。")
                else:
                    st.error(f"エラーが発生しました: {e}")
                return

        if results is None:
            st.error("計算に失敗しました。パネルデータを確認してください。")
            return

        st.session_state["calc_condition"] = condition.strip()
        st.session_state["calc_analysis"]  = analysis
        st.session_state["calc_results"]   = results

        append_history({
            "datetime":    datetime.now().strftime("%Y/%m/%d %H:%M"),
            "condition":   condition.strip(),
            "adj_est":     results["adj_est"],
            "adj_est_min": results["adj_est_min"],
            "adj_inc":     results["adj_inc"],
            "difficulty":  results["difficulty"],
            "analysis":    analysis,
            "results":     results,
        })

    if st.session_state.get("calc_results"):
        st.divider()
        show_results(
            st.session_state["calc_condition"],
            st.session_state["calc_analysis"],
            st.session_state["calc_results"],
        )


# ─────────────────────────────────────────────────────────────────
# ページ②：パネルデータ管理
# ─────────────────────────────────────────────────────────────────

def page_panel_setup():
    st.title("🔧 パネルデータ管理")
    st.caption("パネルデータはCSVで一括更新します。更新がない限り再入力は不要です。")

    panel = load_panel()
    tab_view, tab_update = st.tabs(["📋 現在のパネルデータ", "📤 CSVで更新"])

    with tab_view:
        if panel["total"] == 0:
            st.warning("パネルデータが未設定です。「CSVで更新」タブからアップロードしてください。")
        else:
            st.success(f"✅ パネル総数：**{panel['total']:,}人**　（男性 {panel['gender'].get('男性',0):,}人 / 女性 {panel['gender'].get('女性',0):,}人）")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**年代別**")
                age_df = pd.DataFrame([
                    {"年代": k, "人数": f"{v:,}人"}
                    for k, v in panel["age"].items() if v > 0
                ])
                if not age_df.empty:
                    st.dataframe(age_df, use_container_width=True, hide_index=True)

            with col2:
                if panel.get("prefecture"):
                    st.write("**都道府県別（上位10件）**")
                    top_pref = sorted(panel["prefecture"].items(), key=lambda x: x[1], reverse=True)[:10]
                    pref_df = pd.DataFrame([{"都道府県": k, "人数": f"{v:,}人"} for k, v in top_pref])
                    st.dataframe(pref_df, use_container_width=True, hide_index=True)

            if panel.get("attributes"):
                st.write("**属性カテゴリ一覧**")
                attr_rows = []
                for cat_name, cat_info in panel["attributes"].items():
                    cov = cat_info.get("coverage", 0)
                    reliability = "🟢 完全回収" if cov >= 0.85 else f"🟡 部分回収（{cov*100:.0f}%）"
                    answered = sum(
                        v for k, v in cat_info.get("data", {}).items()
                        if k not in UNKNOWN_KEYS
                    )
                    attr_rows.append({
                        "カテゴリ": cat_name,
                        "選択肢数": f"{len(cat_info.get('data', {}))}個",
                        "回答人数": f"{answered:,}人",
                        "回収状況": reliability,
                    })
                st.dataframe(pd.DataFrame(attr_rows), use_container_width=True, hide_index=True)
                st.caption("🟢完全回収：母数として直接使用　🟡部分回収：参考値として使用")

    with tab_update:
        st.write("モニタスからダウンロードしたCSVをそのままアップロードしてください。")
        st.info(
            "対応フォーマット：モニタス標準CSV（カテゴリ, 属性値, 人数 の列構成）\n"
            "70代/80代/90代以上は「70代以上」に自動統合されます。"
        )

        uploaded = st.file_uploader("CSVファイルを選択", type=["csv"])
        if uploaded:
            try:
                df     = pd.read_csv(uploaded, header=None, encoding="utf-8-sig")
                result = _parse_monitas_csv(df)
                _save_json(PANEL_FILE, result)
                st.success(
                    f"✅ 更新完了！\n\n"
                    f"総数：{result['total']:,}人　"
                    f"男性：{result['gender']['男性']:,}人　"
                    f"女性：{result['gender']['女性']:,}人\n"
                    f"属性カテゴリ：{len(result.get('attributes', {}))}件読み込み"
                )
                st.rerun()
            except Exception as e:
                st.error(f"読み込みエラー: {e}")


# ─────────────────────────────────────────────────────────────────
# ページ③：過去事例検索
# ─────────────────────────────────────────────────────────────────

def page_qa_search():
    st.title("📋 過去事例を検索")
    st.write("過去に試算した回収実績データを辞書として検索できます。")

    qa_hist = load_qa_history()
    if not qa_hist:
        st.warning("事例データが見つかりません。")
        return

    query = st.text_input(
        "キーワードで検索",
        placeholder="例）建設 経営者　／　看護師　／　IT エンジニア　／　投資",
    )

    categories = sorted({r["category"] for r in qa_hist})
    selected_cat = st.selectbox("カテゴリで絞り込み（任意）", ["すべて"] + categories)

    pool = qa_hist if selected_cat == "すべて" else [r for r in qa_hist if r["category"] == selected_cat]

    if query.strip():
        results = search_similar_cases(query.strip(), pool, top_k=20)
    else:
        results = pool[:50]

    st.caption(f"表示件数：{len(results)}件　（全{len(pool)}件）")

    if results:
        rows = []
        for r in results:
            rows.append({
                "カテゴリ": r.get("category", ""),
                "業界": r.get("industry", ""),
                "対象者": r.get("target", "")[:50],
                "回収実績": r.get("n_raw", ""),
                "付加条件": r.get("conditions", "")[:40],
                "備考": r.get("notes", "")[:60],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=500)
    else:
        st.info("該当する事例が見つかりませんでした。")


# ─────────────────────────────────────────────────────────────────
# ページ④：計算履歴
# ─────────────────────────────────────────────────────────────────

def page_history():
    st.title("📁 計算履歴")
    history = load_history()

    if not history:
        st.info("まだ計算履歴がありません。")
        return

    rows = []
    for h in history:
        cond = h.get("condition", "")
        rows.append({
            "日時":         h.get("datetime", ""),
            "条件":         (cond[:35] + "…") if len(cond) > 35 else cond,
            "難易度":       h.get("difficulty", ""),
            "一般的見込み": f"{h.get('adj_est', 0):,}人",
            "固め見込み":   f"{h.get('adj_est_min', 0):,}人",
            "出現率":       f"{h.get('adj_inc', '')}%",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    idx = st.selectbox(
        "詳細を見る（履歴を選択）",
        range(len(history)),
        format_func=lambda i: f"{history[i].get('datetime','')} ─ {history[i].get('condition','')[:45]}…",
    )
    if idx is not None and history[idx].get("results"):
        h = history[idx]
        show_results(h["condition"], h["analysis"], h["results"])

    st.divider()
    if st.button("🗑️ 履歴をすべて削除", type="secondary"):
        st.session_state.session_history = []
        st.success("削除しました")
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# サイドバー＋ルーティング
# ─────────────────────────────────────────────────────────────────

def main():
    inject_css()
    config = load_config()

    # ページ状態の初期化
    if "page" not in st.session_state:
        st.session_state["page"] = "calculation"

    with st.sidebar:
        # ── ロゴ（クリックで新しい調査へ） ──────────────────────
        st.markdown("""
        <div style="padding: 16px 4px 4px 4px;">
            <div style="font-size:11px; color:#5A7A96; letter-spacing:0.08em; margin-bottom:2px;">POWERED BY GEMINI AI</div>
            <div style="font-size:20px; font-weight:800; color:#FFFFFF; letter-spacing:0.01em;">n数試算ツール</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 新しい調査を始める", use_container_width=True, key="title_reset"):
            reset_calc()
            st.session_state["page"] = "calculation"
            st.rerun()

        st.divider()

        # ── API Key ──────────────────────────────────────────────
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""
        if not api_key:
            api_key = st.text_input(
                "🔑 Gemini API Key",
                value=config.get("api_key", ""),
                type="password",
                help="aistudio.google.com から取得できます",
            )
            if api_key != config.get("api_key", ""):
                config["api_key"] = api_key
                save_config(config)

        st.divider()

        # ── ナビゲーション ───────────────────────────────────────
        nav_items = [
            ("calculation", "📊  回収見込み計算"),
            ("history",     "📁  計算履歴"),
            ("qa_search",   "📋  事例検索"),
        ]
        for page_key, label in nav_items:
            btn_type = "primary" if st.session_state["page"] == page_key else "secondary"
            if st.button(label, use_container_width=True, key=f"nav_{page_key}", type=btn_type):
                st.session_state["page"] = page_key
                st.rerun()

        st.divider()

        # ── 管理者設定 ───────────────────────────────────────────
        with st.expander("⚙️ 管理者設定", expanded=False):
            admin_code = st.text_input("管理者コード", type="password", key="admin_code")
            if admin_code == "monitas":
                st.session_state["admin_mode"] = True
                st.success("管理者モードON")

        st.divider()

        # ── パネル状況 ───────────────────────────────────────────
        panel = load_panel()
        if panel["total"] > 0:
            attr_count = len(panel.get("attributes", {}))
            st.success(f"✅ パネル設定済み\n**{panel['total']:,}人**")
            if attr_count > 0:
                st.caption(f"属性 {attr_count}カテゴリ")
        else:
            st.warning("⚠️ パネルデータ未設定")

    # ── ルーティング ─────────────────────────────────────────────
    panel = load_panel()
    current_page = st.session_state.get("page", "calculation")

    if current_page == "calculation":
        page_calculation(api_key, panel)
    elif current_page == "history":
        page_history()
    elif current_page == "qa_search":
        page_qa_search()

    # 管理者モード：パネルデータ管理
    if st.session_state.get("admin_mode"):
        with st.sidebar:
            if st.button("🔧 パネルデータ管理を開く", use_container_width=True):
                st.session_state["show_panel_mgmt"] = True
        if st.session_state.get("show_panel_mgmt"):
            page_panel_setup()


if __name__ == "__main__":
    main()
