"""
モニタス 出現率計算ツール
クライアントの調査条件から回収見込み数・難易度を自動推計します
"""

import streamlit as st
import anthropic
import json
import os
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# 定数・初期設定
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="モニタス 出現率計算ツール",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR     = "data"
PANEL_FILE   = os.path.join(DATA_DIR, "panel_data.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
CONFIG_FILE  = os.path.join(DATA_DIR, "config.json")
MODEL        = "claude-sonnet-4-6"

AGE_GROUPS   = ["10代", "20代", "30代", "40代", "50代", "60代", "70代以上"]
GENDERS      = ["男性", "女性"]
UNKNOWN_KEYS = {"未取得", "わからない", "不明", "無回答"}

os.makedirs(DATA_DIR, exist_ok=True)


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


def load_history() -> list:
    return _load_json(HISTORY_FILE, [])


def load_config() -> dict:
    return _load_json(CONFIG_FILE, {"api_key": ""})


def save_config(cfg: dict):
    _save_json(CONFIG_FILE, cfg)


def append_history(entry: dict):
    history = load_history()
    history.insert(0, entry)
    _save_json(HISTORY_FILE, history[:100])


# ─────────────────────────────────────────────────────────────────
# モニタスCSV パーサー
# ─────────────────────────────────────────────────────────────────

def _parse_monitas_csv(df: pd.DataFrame) -> dict:
    """モニタス母数シートのCSVを解析してパネルデータ辞書を返す"""

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
    """補正後出現率・固め人数 → (ラベル, color, 説明文)"""
    if adj_inc >= 5.0 and adj_est_min >= 500:
        return "易", "success", "回収しやすい条件です。十分な人数が見込めます。"
    elif adj_inc >= 1.5 and adj_est_min >= 200:
        return "普通", "info", "標準的な回収難易度です。"
    elif adj_inc >= 0.4 and adj_est_min >= 50:
        return "難", "warning", "回収に工夫が必要な条件です。緩和措置も検討してください。"
    else:
        return "困難", "error", "回収が非常に困難です。条件の見直しを強く推奨します。"


# ─────────────────────────────────────────────────────────────────
# Claude API：条件分析
# ─────────────────────────────────────────────────────────────────

def analyze_condition(api_key: str, condition: str, panel: dict) -> dict:
    """調査条件をClaudeで分析し、回収見込み・難易度・緩和措置を返す"""

    client = anthropic.Anthropic(api_key=api_key)

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
            lines.append(f"\n【{cat_name}】（{reliability}）")
            for val, cnt in cat_info.get("data", {}).items():
                lines.append(f"  {val}: {cnt:,}人")
    panel_text = "\n".join(lines)

    user_prompt = f"""以下の市場調査ターゲット条件を分析し、モニターパネルからの回収見込みを推計してください。

【調査ターゲット条件】
{condition}

【モニターパネルデータ】
{panel_text}

次のJSON形式のみで回答してください（JSON以外のテキストは不要です）：

{{
  "condition_summary": "条件を1〜2行で要約",
  "include_ages": ["30代", "40代", "50代"],
  "exclude_ages": ["10代", "20代", "60代", "70代以上"],
  "exclude_reason": "除外した理由の説明",
  "gender_specified": false,
  "include_genders": ["男性", "女性"],
  "prefecture_specified": false,
  "include_prefectures": [],
  "attribute_filters": [
    {{
      "category": "カテゴリ名",
      "values": ["属性値1", "属性値2"],
      "is_reliable": true,
      "note": "このフィルタを選んだ理由"
    }}
  ],
  "behavioral_rate": 0.05,
  "behavioral_rate_min": 0.03,
  "behavioral_rate_max": 0.08,
  "behavioral_reasoning": "推計根拠（統計データ・調査名を引用）",
  "confidence": "medium",
  "difficulty_reason": "この条件が難しい・易しい理由を1〜2文で説明",
  "relaxation_suggestions": [
    {{
      "action": "緩和する内容（例: 年代を60代まで拡大）",
      "additional_est": 1500,
      "trade_off": "緩和した場合の条件純度・品質への影響"
    }}
  ],
  "warnings": []
}}

【判断基準】※すべて「保守的・厳しめ」に設定すること

・include_ages: 条件に「確実に」該当する年代のみ。迷ったら除外する。
  - 年代の境界は厳格に。「可能性がある」程度は除外する
  例）NISA投資歴3年以内 → 10代・20代前半・70代以上を除外 → 20代後半〜60代
  例）子育て中 → 10代・20代前半・60代以上を除外 → 20代後半〜50代
  例）会社でのBtoB購買担当 → 10代・20代前半・60代以上を除外 → 20代後半〜50代

・attribute_filters: 条件に「直接かつ明確に」関連するカテゴリのみ。迷ったら含めない。
  - 「間違いなく当てはまる」値のみ。周辺的な値は含めない
  - 例）IT業界勤務 → 業種=ソフトウェア業・情報サービス業のみ（電気通信業等は含めない）
  - 例）正社員会社員 → 職業=会社員3種のみ（自営業・パートは含めない）

・behavioral_rate: 必ず保守的な値に設定すること
  - 属性合致でも詳細スクリーニングで外れるケースが多い
  - 純粋な属性条件のみでも最大0.80に抑えること（申告と実態の乖離を考慮）
  - 行動・経験条件がある場合は0.05〜0.30程度

・relaxation_suggestions: 2〜3件、現実的で具体的な緩和案を提示すること
  - additional_est はパネルデータの実数と行動率を踏まえたAI推計値（概算）
  - 条件の「核」を保ちつつ、周辺を緩和するアイデアを提案する
  - 緩和した場合のデメリット・純度への影響を必ずtrade_offに記載する

・confidence: high（公的統計を具体的に引用可能）/ medium（業界推計あり）/ low（根拠が薄い）
・性別・地域が条件に明示されていなければ specified = false
"""

    res = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system="あなたは市場調査・パネル調査の専門家です。必ず有効なJSONのみを返してください。",
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = res.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.split("```")[0]
    return json.loads(text.strip())


# ─────────────────────────────────────────────────────────────────
# 計算ロジック
# ─────────────────────────────────────────────────────────────────

def calculate(panel: dict, analysis: dict, activity_rate: float = 0.6) -> dict | None:
    """回収見込み人数・難易度を計算する（目標n数不要）"""

    total = panel["total"]
    if total == 0:
        return None

    # ① 年代ベース
    include_ages = analysis.get("include_ages", AGE_GROUPS)
    age_base = sum(panel["age"].get(a, 0) for a in include_ages)

    # ② 性別調整
    gender_ratio = 1.0
    if analysis.get("gender_specified"):
        genders  = analysis.get("include_genders", GENDERS)
        g_count  = sum(panel["gender"].get(g, 0) for g in genders)
        gender_ratio = g_count / total if total > 0 else 1.0

    # ③ 都道府県調整
    pref_ratio = 1.0
    if analysis.get("prefecture_specified") and analysis.get("include_prefectures"):
        prefs = analysis["include_prefectures"]
        if panel.get("prefecture"):
            p_count   = sum(panel["prefecture"].get(p, 0) for p in prefs)
            pref_ratio = p_count / total if total > 0 else 1.0

    # ④ 追加属性フィルタ
    attr_filter_details = []
    attr_combined_ratio = 1.0
    for af in analysis.get("attribute_filters", []):
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
        attr_filter_details.append({
            "category": cat_name,
            "values":   values,
            "matched":  matched,
            "ratio":    round(ratio * 100, 2),
            "coverage": coverage,
            "reliable": af.get("is_reliable", coverage >= 0.85),
            "note":     af.get("note", ""),
        })

    # ⑤ 人口学的ベース（独立性仮定）
    age_ratio = age_base / total if total > 0 else 1.0
    demo_base = total * age_ratio * gender_ratio * pref_ratio * attr_combined_ratio

    # ⑥ 行動・態度条件
    brate     = min(float(analysis.get("behavioral_rate",     1.0)), 1.0)
    brate_min = min(float(analysis.get("behavioral_rate_min", brate * 0.6)), 1.0)
    brate_max = min(float(analysis.get("behavioral_rate_max", brate * 1.4)), 1.0)

    est     = demo_base * brate
    est_min = demo_base * brate_min
    est_max = demo_base * brate_max

    # ⑦ 回収率補正（パネル稼働率）
    adj_est     = est     * activity_rate
    adj_est_min = est_min * activity_rate
    adj_est_max = est_max * activity_rate

    adj_inc     = adj_est     / total * 100
    adj_inc_min = adj_est_min / total * 100
    adj_inc_max = adj_est_max / total * 100

    # ⑧ 難易度判定
    difficulty, diff_color, diff_desc = _difficulty(adj_inc, int(adj_est_min))

    return {
        "total":               total,
        "age_base":            int(age_base),
        "demo_base":           int(demo_base),
        "gender_ratio":        gender_ratio,
        "pref_ratio":          pref_ratio,
        "attr_filter_details": attr_filter_details,
        "include_ages":        include_ages,
        "exclude_ages":        analysis.get("exclude_ages", []),
        "behavioral_rate":     brate,
        # 理論値
        "est":                 int(est),
        "est_min":             int(est_min),
        "est_max":             int(est_max),
        # 実回収見込み（補正後）
        "activity_rate":       activity_rate,
        "adj_est":             int(adj_est),       # 一般的
        "adj_est_min":         int(adj_est_min),   # 固め
        "adj_est_max":         int(adj_est_max),
        "adj_inc":             round(adj_inc,     2),
        "adj_inc_min":         round(adj_inc_min, 2),
        "adj_inc_max":         round(adj_inc_max, 2),
        # 難易度
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
        f"  ※パネル稼働率補正：{r['activity_rate']*100:.0f}%",
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

def show_results(condition: str, analysis: dict, r: dict):

    st.divider()

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

    # ── 緩和措置アドバイス ───────────────────────────────────────
    suggestions = analysis.get("relaxation_suggestions", [])
    if suggestions:
        st.divider()
        st.write("**💡 n数を増やしたい場合の緩和措置**")
        st.caption("条件を少し変えた場合の追加回収見込み（AI推計・概算値）")
        for s in suggestions:
            action      = s.get("action", "")
            additional  = s.get("additional_est", 0)
            trade_off   = s.get("trade_off", "")
            with st.container(border=True):
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
        st.warning("⚠️ パネルデータが未設定です。「🔧 パネルデータ管理」からCSVをアップロードしてください。")
        return

    condition = st.text_area(
        "調査ターゲット条件",
        placeholder=(
            "例：新NISAなど長期の放置型運用をしている投資歴3年以内の方\n"
            "例：小学生以下の子どもがいる共働き世帯の母親"
        ),
        height=110,
    )

    with st.expander("⚙️ 詳細設定（通常はデフォルトのままでOK）"):
        activity_pct = st.slider(
            "回収率補正 (%)",
            min_value=30, max_value=90, value=60, step=5,
            help=(
                "パネル稼働率 × スクリーニング完了率の想定値。\n"
                "難易度が高い条件（特定職種・行動経験者など）は低めに設定。\n"
                "デフォルト60%＝業界の保守的な標準値。"
            ),
        )
        activity_rate = activity_pct / 100

    if not api_key:
        st.warning("サイドバーでClaude API Keyを設定してください。")

    if st.button(
        "🔍 回収見込みを計算する",
        type="primary",
        disabled=(not condition.strip() or not api_key),
    ):
        with st.spinner("Claudeが分析中... （10〜20秒かかります）"):
            try:
                analysis = analyze_condition(api_key, condition.strip(), panel)
                results  = calculate(panel, analysis, activity_rate)
            except json.JSONDecodeError:
                st.error("AIの応答を解析できませんでした。もう一度お試しください。")
                return
            except Exception as e:
                if "API_KEY_INVALID" in str(e) or "invalid" in str(e).lower():
                    st.error("APIキーが無効です。サイドバーで正しいAPIキーを入力してください。")
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
            "datetime":   datetime.now().strftime("%Y/%m/%d %H:%M"),
            "condition":  condition.strip(),
            "adj_est":    results["adj_est"],
            "adj_est_min": results["adj_est_min"],
            "adj_inc":    results["adj_inc"],
            "difficulty": results["difficulty"],
            "analysis":   analysis,
            "results":    results,
        })

    if st.session_state.get("calc_results"):
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

    # ── 現在のデータ表示（読み取り専用） ────────────────────────
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

    # ── CSV更新 ──────────────────────────────────────────────────
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
# ページ③：計算履歴
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
        _save_json(HISTORY_FILE, [])
        st.success("削除しました")
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# サイドバー＋ルーティング
# ─────────────────────────────────────────────────────────────────

def main():
    config = load_config()

    with st.sidebar:
        st.title("📊 モニタス\n出現率計算ツール")
        st.divider()

        if "CLAUDE_API_KEY" in st.secrets:
            api_key = st.secrets["CLAUDE_API_KEY"]
            st.success("✅ APIキー設定済み")
        else:
            api_key = st.text_input(
                "🔑 Claude API Key",
                value=config.get("api_key", ""),
                type="password",
                help="console.anthropic.com から取得できます",
            )
            if api_key != config.get("api_key", ""):
                config["api_key"] = api_key
                save_config(config)

        st.divider()

        page = st.radio(
            "メニュー",
            ["📊 回収見込み計算", "📁 計算履歴", "🔧 パネルデータ管理"],
        )

        st.divider()
        panel = load_panel()
        if panel["total"] > 0:
            attr_count = len(panel.get("attributes", {}))
            st.success(f"✅ パネル設定済み\n**{panel['total']:,}人**")
            if attr_count > 0:
                st.caption(f"属性 {attr_count}カテゴリ")
        else:
            st.warning("⚠️ パネルデータ未設定")

    panel = load_panel()
    if page == "📊 回収見込み計算":
        page_calculation(api_key, panel)
    elif page == "📁 計算履歴":
        page_history()
    else:
        page_panel_setup()


if __name__ == "__main__":
    main()
