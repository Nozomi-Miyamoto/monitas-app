"""
モニタス 出現率計算ツール
クライアントの調査条件から出現率・必要スクリーニング数を自動推計します
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

AGE_GROUPS  = ["10代", "20代", "30代", "40代", "50代", "60代", "70代以上"]
GENDERS     = ["男性", "女性"]
PREFECTURES = [
    "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県",
    "茨城県","栃木県","群馬県","埼玉県","千葉県","東京都","神奈川県",
    "新潟県","富山県","石川県","福井県","山梨県","長野県","岐阜県",
    "静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県",
    "奈良県","和歌山県","鳥取県","島根県","岡山県","広島県","山口県",
    "徳島県","香川県","愛媛県","高知県","福岡県","佐賀県","長崎県",
    "熊本県","大分県","宮崎県","鹿児島県","沖縄県",
]

# カバレッジ計算から除外する「未回答」扱いのキー
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
        "total":      0,
        "age":        {a: 0 for a in AGE_GROUPS},
        "gender":     {"男性": 0, "女性": 0},
        "prefecture": {},
        "attributes": {},   # 追加属性（未既婚・職業・業種・年収など）
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
    """モニタス母数シートのCSVを解析してパネルデータ辞書を返す。

    対応フォーマット（列構成）:
      カテゴリ, 属性値, 人数, [男性, 人数, 女性, 人数]
    カテゴリ列が空の場合は直前のカテゴリを引き継ぐ。
    70代・80代・90代以上は「70代以上」に自動統合する。
    未取得・わからない等はカバレッジ計算から除外して信頼度を判定する。
    """

    def to_int(val) -> int:
        if pd.isna(val):
            return 0
        s = str(val).replace(",", "").strip()
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return 0

    gender: dict     = {"男性": 0, "女性": 0}
    age_raw: dict    = {}          # 生の年代データ（70代統合前）
    prefecture: dict = {}          # 都道府県 → 人数
    raw_attrs: dict  = {}          # その他カテゴリ → {値: 人数}

    current_category: str | None = None

    for _, row in df.iterrows():
        cells = [str(v).strip() if not pd.isna(v) else "" for v in row]
        while len(cells) < 7:
            cells.append("")

        col0, col1, col2 = cells[0], cells[1], cells[2]

        # 完全空行はスキップ
        if not col0 and not col1:
            continue

        # カテゴリ列が空でなければ更新
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
            # その他すべてのカテゴリを汎用的に取り込む
            if current_category not in raw_attrs:
                raw_attrs[current_category] = {}
            raw_attrs[current_category][col1] = count

    # 年代：70代以上に統合
    age = {a: 0 for a in AGE_GROUPS}
    age_70plus = 0
    for age_name, cnt in age_raw.items():
        if age_name in ("70代", "80代", "90代以上"):
            age_70plus += cnt
        elif age_name in AGE_GROUPS:
            age[age_name] = cnt
    age["70代以上"] = age_70plus

    # 総数（性別合計を正とし、取れない場合は年代合計を使う）
    total = gender["男性"] + gender["女性"]
    if total == 0:
        total = sum(age.values())

    # その他カテゴリ：カバレッジ計算
    # coverage = (未取得・わからないを除いた回答数) / パネル総数
    attributes: dict = {}
    for cat_name, cat_data in raw_attrs.items():
        answered = sum(v for k, v in cat_data.items() if k not in UNKNOWN_KEYS)
        coverage = round(min(answered / total, 1.0), 3) if total > 0 else 0.0
        attributes[cat_name] = {
            "data":     cat_data,
            "coverage": coverage,
        }

    return {
        "total":      total,
        "age":        age,
        "gender":     gender,
        "prefecture": prefecture,
        "attributes": attributes,
    }


# ─────────────────────────────────────────────────────────────────
# Claude API：条件分析
# ─────────────────────────────────────────────────────────────────

def analyze_condition(api_key: str, condition: str, panel: dict) -> dict:
    """調査条件をClaudeで分析し、対象年代・属性フィルタ・出現率推計などを構造化して返す"""

    client = anthropic.Anthropic(api_key=api_key)

    # ── パネルデータをテキスト化 ───────────────────────────────────
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

    # 追加属性カテゴリ（カバレッジ付き）
    if panel.get("attributes"):
        for cat_name, cat_info in panel["attributes"].items():
            cov = cat_info.get("coverage", 0)
            reliability = "完全回収" if cov >= 0.85 else f"部分回収（回答率{cov*100:.0f}%）"
            lines.append(f"\n【{cat_name}】（{reliability}）")
            for val, cnt in cat_info.get("data", {}).items():
                lines.append(f"  {val}: {cnt:,}人")

    panel_text = "\n".join(lines)

    user_prompt = f"""以下の市場調査ターゲット条件を分析し、モニターパネルの出現率を推計してください。

【調査ターゲット条件】
{condition}

【モニターパネルデータ】
{panel_text}

次のJSON形式のみで回答してください（JSON以外のテキストは不要です）：

{{
  "condition_summary": "条件を1〜2行で要約",
  "include_ages": ["20代", "30代", "40代", "50代"],
  "exclude_ages": ["10代", "60代", "70代以上"],
  "exclude_reason": "除外した理由の説明",
  "gender_specified": false,
  "include_genders": ["男性", "女性"],
  "prefecture_specified": false,
  "include_prefectures": [],
  "attribute_filters": [
    {{
      "category": "カテゴリ名（例: 未既婚、職業、業種）",
      "values": ["該当する属性値1", "該当する属性値2"],
      "is_reliable": true,
      "note": "このフィルタを選んだ理由・注意事項"
    }}
  ],
  "has_behavioral_condition": true,
  "behavioral_rate": 1.0,
  "behavioral_rate_min": 1.0,
  "behavioral_rate_max": 1.0,
  "behavioral_reasoning": "推計根拠（具体的な統計データ・調査名を引用）",
  "confidence": "medium",
  "warnings": []
}}

【判断基準】
・include_ages: この条件に現実的に該当しうる年代のみ含める
  例）NISA投資 → 10代を除外、20〜60代を含める
  例）子育て中 → 60代以上を除外、20代後半〜50代を含める
  例）介護経験者 → 10〜30代を除外、40〜70代以上を含める

・attribute_filters: パネルデータに存在するカテゴリで条件に直接関連するものを設定する
  - パネルデータに実際に存在するカテゴリ名・属性値のみを使う（存在しない値は使わない）
  - 完全回収のカテゴリ（is_reliable=true）: 未既婚・子有無・職業・業種（勤めていない含む）・個人年収・世帯年収 など
    → coverage 0.85以上のものは完全回収とみなしてよい
  - 部分回収のカテゴリ（is_reliable=false）: 職種・役職・雇用形態・会社の売上規模・職場の規模 など
    → 「未取得」が多く、実際の人数はこれより多い可能性がある。noteに明記する
  - 条件に明示されていない属性は追加しない（過剰フィルタ禁止）
  - 属性フィルタで条件の人口学的特性が完全に特定できる場合は behavioral_rate=1.0 でよい

・behavioral_rate: attribute_filtersで絞った後の母数に対して、さらに行動・態度条件に該当する割合（0〜1）
  - 属性フィルタ＋年代で既に対象を絞り切れる場合 → behavioral_rate=1.0
  - 「経験あり」「利用中」「購入意向あり」など行動・態度が残る場合 → 統計的根拠に基づき推計

・confidence: high（信頼できる統計データあり）/ medium（一般的な推計）/ low（不確実性高い）
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

def calculate(panel: dict, analysis: dict, target_n: int) -> dict | None:
    """出現率・推定対象人数・必要スクリーニング数を計算する"""

    total = panel["total"]
    if total == 0:
        return None

    # ① 年代ベースの集計
    include_ages = analysis.get("include_ages", AGE_GROUPS)
    age_base = sum(panel["age"].get(a, 0) for a in include_ages)

    # ② 性別調整（条件に性別指定がある場合のみ）
    gender_ratio = 1.0
    if analysis.get("gender_specified"):
        genders  = analysis.get("include_genders", GENDERS)
        g_count  = sum(panel["gender"].get(g, 0) for g in genders)
        gender_ratio = g_count / total if total > 0 else 1.0

    # ③ 都道府県調整（条件に地域指定がある場合のみ）
    pref_ratio = 1.0
    if analysis.get("prefecture_specified") and analysis.get("include_prefectures"):
        prefs = analysis["include_prefectures"]
        if panel.get("prefecture"):
            p_count   = sum(panel["prefecture"].get(p, 0) for p in prefs)
            pref_ratio = p_count / total if total > 0 else 1.0

    # ④ 追加属性フィルタ（未既婚・職業・業種・年収など）
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

    # ⑤ 人口学的ベース人数（全フィルタを独立性仮定で掛け合わせ）
    # demo_base = total × age_ratio × gender_ratio × pref_ratio × 各属性ratio
    age_ratio = age_base / total if total > 0 else 1.0
    demo_base = total * age_ratio * gender_ratio * pref_ratio * attr_combined_ratio

    # ⑥ 行動・態度条件の出現率を適用
    brate     = min(float(analysis.get("behavioral_rate",     1.0)), 1.0)
    brate_min = min(float(analysis.get("behavioral_rate_min", brate * 0.6)), 1.0)
    brate_max = min(float(analysis.get("behavioral_rate_max", brate * 1.4)), 1.0)

    est     = demo_base * brate
    est_min = demo_base * brate_min
    est_max = demo_base * brate_max

    # ⑦ 全パネルに対する出現率（%）
    inc     = est     / total * 100
    inc_min = est_min / total * 100
    inc_max = est_max / total * 100

    # ⑧ 必要スクリーニング数
    def req_ss(inc_pct: float) -> int:
        return int(target_n / (inc_pct / 100)) if inc_pct > 0 else 0

    # ⑨ 実現可能性判定
    if est_min >= target_n * 3:
        feasibility, fstatus = "達成見込み十分", "success"
    elif est_min >= target_n:
        feasibility, fstatus = "達成可能（余裕少）", "warning"
    elif est_max >= target_n:
        feasibility, fstatus = "推計次第で達成可能", "warning"
    else:
        feasibility, fstatus = "n数不足リスクあり", "error"

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
        "est":                 int(est),
        "est_min":             int(est_min),
        "est_max":             int(est_max),
        "inc":                 round(inc,     2),
        "inc_min":             round(inc_min, 2),
        "inc_max":             round(inc_max, 2),
        "req_ss":              req_ss(inc),
        "req_ss_conservative": req_ss(inc_min),
        "feasibility":         feasibility,
        "fstatus":             fstatus,
        "target_n":            target_n,
    }


def make_report(condition: str, analysis: dict, r: dict) -> str:
    """クライアント共有用テキストレポートを生成する"""
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    sep = "━" * 42
    lines = [
        sep,
        "  市場調査 出現率推計レポート",
        f"  作成日時：{now}",
        sep,
        "",
        "■ 調査ターゲット条件",
        f"  {condition}",
        "",
        "■ 条件の解釈",
        f"  {analysis.get('condition_summary', '')}",
        "",
        "■ 対象母数の算出",
        f"  パネル総数          ：{r['total']:,}人",
        f"  対象年代            ：{' / '.join(r['include_ages'])}",
        f"  除外年代            ：{' / '.join(r['exclude_ages'])}",
        f"    └ 理由：{analysis.get('exclude_reason', '')}",
        f"  年代ベース人数      ：{r['age_base']:,}人",
    ]

    if r.get("attr_filter_details"):
        lines.append("  属性フィルタ（絞り込み）：")
        for af in r["attr_filter_details"]:
            tag = "完全回収" if af.get("reliable") else f"部分回収（回答率{af['coverage']*100:.0f}%）"
            lines.append(
                f"    ・{af['category']}＝{' / '.join(af['values'])}"
                f"　{af['matched']:,}人（全体の{af['ratio']}%）[{tag}]"
            )

    lines += [
        f"  調整後母数          ：{r['demo_base']:,}人",
        "",
        "■ 出現率推計",
        f"  行動・態度条件の出現率：約 {r['behavioral_rate']*100:.1f}%",
        f"  推定出現率（中央値）  ：{r['inc']}%",
        f"  推定出現率（幅）      ：{r['inc_min']}% 〜 {r['inc_max']}%",
        f"  推定対象人数          ：約 {r['est']:,}人",
        f"    └ 幅：{r['est_min']:,} 〜 {r['est_max']:,}人",
        "",
        "■ n数・スクリーニング数",
        f"  目標n数                     ：{r['target_n']:,}人",
        f"  必要スクリーニング数（標準）：{r['req_ss']:,} ss",
        f"  必要スクリーニング数（保守）：{r['req_ss_conservative']:,} ss",
        "",
        f"■ 実現可能性：{r['feasibility']}",
        "",
        "■ 推計根拠",
        f"  {analysis.get('behavioral_reasoning', '')}",
        "",
        "※ 本推計はAIによる統計的推定値です。",
        "  実際の出現率はスクリーニング調査での確認を推奨します。",
        sep,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 共通UIコンポーネント：結果表示
# ─────────────────────────────────────────────────────────────────

def show_results(condition: str, analysis: dict, r: dict):
    """計算結果をStreamlit上に表示する"""

    st.divider()
    st.subheader("📋 分析結果")
    st.info(f"**条件の解釈：** {analysis.get('condition_summary', '')}")

    # 対象・除外年代
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**✅ 対象年代**")
        st.write("　" + " ／ ".join(r["include_ages"]))
    with col_b:
        if r["exclude_ages"]:
            st.write("**❌ 除外年代**")
            st.write("　" + " ／ ".join(r["exclude_ages"]))
            st.caption(analysis.get("exclude_reason", ""))

    # 属性フィルタ表示
    if r.get("attr_filter_details"):
        st.write("**🔍 属性フィルタ（パネルデータから直接絞り込み）**")
        for af in r["attr_filter_details"]:
            if af.get("reliable", True):
                tag = "🟢 完全回収"
            else:
                tag = f"🟡 部分回収（回答率{af['coverage']*100:.0f}%）"
            st.write(
                f"　・**{af['category']}**：{' / '.join(af['values'])}"
                f"　→ {af['matched']:,}人（全体の{af['ratio']}%）　{tag}"
            )
            if not af.get("reliable", True):
                st.caption(
                    "　　⚠️ このカテゴリは全員が回答しているわけではないため、"
                    "実際の対象人数はこれより多い可能性があります。"
                )

    st.divider()

    # 4つのメインメトリクス
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "推定対象人数",
        f"{r['est']:,}人",
        f"幅：{r['est_min']:,} 〜 {r['est_max']:,}人",
    )
    m2.metric(
        "推定出現率",
        f"{r['inc']}%",
        f"幅：{r['inc_min']} 〜 {r['inc_max']}%",
    )
    m3.metric("必要SS数（標準）", f"{r['req_ss']:,} ss")
    m4.metric("必要SS数（保守）", f"{r['req_ss_conservative']:,} ss")

    # 実現可能性バナー
    emoji_map = {"success": "✅", "warning": "⚠️", "error": "❌"}
    banner = f"{emoji_map.get(r['fstatus'], '')} 実現可能性：{r['feasibility']}"
    if r["fstatus"] == "success":
        st.success(banner)
    elif r["fstatus"] == "warning":
        st.warning(banner)
    else:
        st.error(banner)

    # 推計根拠・計算詳細（折りたたみ）
    with st.expander("📖 推計根拠・計算の詳細を見る"):
        st.write("**推計根拠**")
        st.write(analysis.get("behavioral_reasoning", ""))

        conf_map = {
            "high":   "🟢 高（信頼できる統計データあり）",
            "medium": "🟡 中（一般的な推計）",
            "low":    "🔴 低（不確実性が高い）",
        }
        st.write(f"**推計信頼度：** {conf_map.get(analysis.get('confidence', 'medium'), '')}")

        if analysis.get("warnings"):
            st.warning("**注意事項：**\n" + "\n".join(f"• {w}" for w in analysis["warnings"]))

        st.write("**計算の内訳**")
        st.write(f"• パネル総数：{r['total']:,}人")
        st.write(f"• 年代ベース（{' / '.join(r['include_ages'])}）：{r['age_base']:,}人")
        if r["gender_ratio"] < 1.0:
            st.write(f"• 性別調整（×{r['gender_ratio']:.2f}）")
        if r["pref_ratio"] < 1.0:
            st.write(f"• 地域調整（×{r['pref_ratio']:.2f}）")
        for af in r.get("attr_filter_details", []):
            st.write(f"• {af['category']}フィルタ（×{af['ratio']/100:.4f} ＝ {af['matched']:,}人/{r['total']:,}人）")
        st.write(f"• → 調整後母数：{r['demo_base']:,}人")
        if r["behavioral_rate"] < 1.0:
            st.write(f"• 行動・態度条件の出現率：{r['behavioral_rate']*100:.1f}%")
        st.write(f"• 最終推定対象人数：{r['est']:,}人")

    # クライアント共有用レポート
    st.divider()
    report_text = make_report(condition, analysis, r)
    st.text_area(
        "📋 クライアント共有用レポート（そのままコピーしてご使用ください）",
        report_text,
        height=320,
    )


# ─────────────────────────────────────────────────────────────────
# ページ①：出現率計算（メイン機能）
# ─────────────────────────────────────────────────────────────────

def page_calculation(api_key: str, panel: dict):
    st.title("📊 出現率計算")

    if panel["total"] == 0:
        st.warning("⚠️ パネルデータが未設定です。左メニューの「⚙️ パネルデータ設定」から先にデータを入力してください。")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        condition = st.text_area(
            "調査ターゲット条件",
            placeholder=(
                "例：ここに表示したい条件例を入力してください\n"
                "例：2つ目の例文をここに入力"
            ),
            height=110,
        )
    with col2:
        target_n = st.number_input("目標n数（人）", min_value=1, value=300, step=10)
        st.metric("パネル総数", f"{panel['total']:,}人")

    if not api_key:
        st.warning("サイドバーでClaude API Keyを設定してください。")

    if st.button(
        "🔍 出現率を計算する",
        type="primary",
        disabled=(not condition.strip() or not api_key),
    ):
        with st.spinner("Claudeが分析中... （10〜20秒かかります）"):
            try:
                analysis = analyze_condition(api_key, condition.strip(), panel)
                results  = calculate(panel, analysis, target_n)
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
            st.error("計算に失敗しました。パネルデータの総数を確認してください。")
            return

        st.session_state["calc_condition"] = condition.strip()
        st.session_state["calc_analysis"]  = analysis
        st.session_state["calc_results"]   = results

        append_history({
            "datetime":    datetime.now().strftime("%Y/%m/%d %H:%M"),
            "condition":   condition.strip(),
            "target_n":    target_n,
            "inc":         results["inc"],
            "est":         results["est"],
            "req_ss":      results["req_ss"],
            "feasibility": results["feasibility"],
            "analysis":    analysis,
            "results":     results,
        })

    if st.session_state.get("calc_results"):
        show_results(
            st.session_state["calc_condition"],
            st.session_state["calc_analysis"],
            st.session_state["calc_results"],
        )


# ─────────────────────────────────────────────────────────────────
# ページ②：パネルデータ設定
# ─────────────────────────────────────────────────────────────────

def page_panel_setup():
    st.title("⚙️ パネルデータ設定")
    st.write("モニタスから取り寄せた母数データを入力してください。一度設定すれば変更があるまで再入力不要です。")

    panel = load_panel()
    tab_basic, tab_pref, tab_csv = st.tabs(["基本設定", "都道府県別（任意）", "CSVで一括入力"])

    # ── 基本設定 ──────────────────────────────────────────────────
    with tab_basic:
        st.subheader("パネル総数")
        total = st.number_input("総モニター数（人）", min_value=0, value=panel["total"], step=1000)

        st.subheader("年代別人数")
        age_vals = {}
        cols = st.columns(len(AGE_GROUPS))
        for i, age in enumerate(AGE_GROUPS):
            with cols[i]:
                age_vals[age] = st.number_input(
                    age, min_value=0, value=panel["age"].get(age, 0), step=100, key=f"age_{age}"
                )

        age_sum = sum(age_vals.values())
        if total > 0 and age_sum > 0:
            st.caption(f"年代合計：{age_sum:,}人 ／ 総数：{total:,}人")
            if abs(age_sum - total) / total > 0.05:
                st.warning("年代合計と総数の差が5%以上あります。数値を確認してください。")

        st.subheader("性別人数")
        gc1, gc2 = st.columns(2)
        with gc1:
            male   = st.number_input("男性（人）", min_value=0, value=panel["gender"].get("男性", 0), step=100)
        with gc2:
            female = st.number_input("女性（人）", min_value=0, value=panel["gender"].get("女性", 0), step=100)

        if st.button("💾 保存する", type="primary"):
            _save_json(PANEL_FILE, {
                "total":      total,
                "age":        age_vals,
                "gender":     {"男性": male, "女性": female},
                "prefecture": panel.get("prefecture", {}),
                "attributes": panel.get("attributes", {}),
            })
            st.success("✅ 保存しました！")
            st.rerun()

    # ── 都道府県別 ─────────────────────────────────────────────────
    with tab_pref:
        st.subheader("都道府県別人数")
        st.caption("地域指定の条件が多い場合に入力すると計算精度が上がります。入力は任意です。")
        pref_data = panel.get("prefecture", {})
        new_pref  = {}
        cols3 = st.columns(3)
        for i, pref in enumerate(PREFECTURES):
            with cols3[i % 3]:
                new_pref[pref] = st.number_input(
                    pref, min_value=0, value=pref_data.get(pref, 0), step=10, key=f"pref_{pref}"
                )

        if st.button("💾 都道府県データを保存", type="primary", key="save_pref"):
            current = load_panel()
            current["prefecture"] = {k: v for k, v in new_pref.items() if v > 0}
            _save_json(PANEL_FILE, current)
            st.success("✅ 保存しました！")

    # ── CSVインポート ──────────────────────────────────────────────
    with tab_csv:
        st.subheader("モニタスCSVをそのままアップロード")
        st.write("モニタスからダウンロードしたCSVファイルをそのままアップロードしてください。")
        st.info(
            "**対応フォーマット**：モニタス標準フォーマット（カテゴリ, 属性値, 人数 の列構成）\n\n"
            "自動読み込み：性別・年代・都道府県・未既婚・子有無・職業・業種・年収・職種・役職など\n"
            "70代/80代/90代以上は「70代以上」に自動統合されます。"
        )

        uploaded = st.file_uploader("CSVファイルを選択", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded, header=None, encoding="utf-8-sig")
                result = _parse_monitas_csv(df)
                _save_json(PANEL_FILE, result)

                st.success(
                    f"✅ 読み込み完了！\n\n"
                    f"総数：{result['total']:,}人　"
                    f"男性：{result['gender']['男性']:,}人　"
                    f"女性：{result['gender']['女性']:,}人"
                )

                # 年代別プレビュー
                st.write("**年代別**")
                age_df = pd.DataFrame([
                    {"年代": k, "人数": f"{v:,}人"}
                    for k, v in result["age"].items() if v > 0
                ])
                if not age_df.empty:
                    st.dataframe(age_df, use_container_width=True, hide_index=True)

                # 都道府県プレビュー（上位10件）
                if result.get("prefecture"):
                    st.write("**都道府県別（上位10件）**")
                    top_pref = sorted(
                        result["prefecture"].items(), key=lambda x: x[1], reverse=True
                    )[:10]
                    pref_df = pd.DataFrame([
                        {"都道府県": k, "人数": f"{v:,}人"} for k, v in top_pref
                    ])
                    st.dataframe(pref_df, use_container_width=True, hide_index=True)

                # 追加属性カテゴリ一覧
                if result.get("attributes"):
                    st.write("**読み込まれた追加属性カテゴリ**")
                    attr_rows = []
                    for cat_name, cat_info in result["attributes"].items():
                        cov = cat_info.get("coverage", 0)
                        reliability = "🟢 完全回収" if cov >= 0.85 else f"🟡 部分回収（回答率{cov*100:.0f}%）"
                        val_count = len(cat_info.get("data", {}))
                        answered_total = sum(
                            v for k, v in cat_info.get("data", {}).items()
                            if k not in UNKNOWN_KEYS
                        )
                        attr_rows.append({
                            "カテゴリ": cat_name,
                            "選択肢数": f"{val_count}個",
                            "回答人数": f"{answered_total:,}人",
                            "回収状況": reliability,
                        })
                    st.dataframe(
                        pd.DataFrame(attr_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "🟢完全回収：母数として直接使用　"
                        "🟡部分回収：参考値として使用（実際の人数はより多い可能性あり）"
                    )

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
            "日時":     h.get("datetime", ""),
            "条件":     (cond[:38] + "…") if len(cond) > 38 else cond,
            "目標n":    h.get("target_n", ""),
            "出現率":   f"{h.get('inc', '')}%",
            "推定対象": f"{h.get('est', 0):,}人",
            "必要SS":   f"{h.get('req_ss', 0):,}ss",
            "判定":     h.get("feasibility", ""),
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
            ["📊 出現率計算", "⚙️ パネルデータ設定", "📁 計算履歴"],
        )

        st.divider()

        panel = load_panel()
        if panel["total"] > 0:
            attr_count = len(panel.get("attributes", {}))
            st.success(f"✅ パネル設定済み\n**{panel['total']:,}人**")
            if attr_count > 0:
                st.caption(f"追加属性：{attr_count}カテゴリ読み込み済み")
        else:
            st.warning("⚠️ パネルデータ未設定\nまずパネルデータを入力してください")

    panel = load_panel()
    if page == "📊 出現率計算":
        page_calculation(api_key, panel)
    elif page == "⚙️ パネルデータ設定":
        page_panel_setup()
    else:
        page_history()


if __name__ == "__main__":
    main()
