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

DATA_DIR    = "data"
PANEL_FILE  = os.path.join(DATA_DIR, "panel_data.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
MODEL       = "claude-sonnet-4-6"

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
    _save_json(HISTORY_FILE, history[:100])  # 最新100件を保持


# ─────────────────────────────────────────────────────────────────
# Claude API：条件分析
# ─────────────────────────────────────────────────────────────────

def analyze_condition(api_key: str, condition: str, panel: dict) -> dict:
    """調査条件をClaudeで分析し、対象年代・出現率推計などを構造化して返す"""

    client = anthropic.Anthropic(api_key=api_key)

    # パネルデータをテキスト化
    lines = [f"パネル総数: {panel['total']:,}人", "", "【年代別】"]
    for age, n in panel["age"].items():
        lines.append(f"  {age}: {n:,}人")
    lines += ["", "【性別】"]
    for g, n in panel["gender"].items():
        lines.append(f"  {g}: {n:,}人")
    if panel.get("prefecture"):
        top = sorted(panel["prefecture"].items(), key=lambda x: x[1], reverse=True)[:10]
        lines += ["", "【都道府県別（上位10件）】"]
        for p, n in top:
            lines.append(f"  {p}: {n:,}人")
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
  "has_behavioral_condition": true,
  "behavioral_rate": 0.05,
  "behavioral_rate_min": 0.03,
  "behavioral_rate_max": 0.08,
  "behavioral_reasoning": "推計根拠（具体的な統計データ・調査名を引用）",
  "confidence": "medium",
  "warnings": []
}}

【判断基準】
・include_ages: この条件に現実的に該当しうる年代のみ含める
  例）NISA投資 → 10代を除外、20〜60代を含める
  例）子育て中 → 60代以上を除外、20代後半〜50代を含める
  例）介護経験者 → 10〜30代を除外、40〜70代を含める
・behavioral_rate: include_agesの合計人数のうち行動・態度条件に該当する割合（0〜1）
・behavioral_rate_min/max: 推計の保守的〜楽観的な幅
・confidence: high（信頼できる統計データあり）/ medium（一般的な推計）/ low（不確実性高い）
・性別・地域が条件に明示されていなければ specified = false
"""

    res = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system="あなたは市場調査・パネル調査の専門家です。必ず有効なJSONのみを返してください。",
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = res.content[0].text.strip()
    # コードブロックが含まれていた場合に除去
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

    # ④ 人口学的ベース人数
    demo_base = age_base * gender_ratio * pref_ratio

    # ⑤ 行動・態度条件の出現率を適用
    brate     = min(float(analysis.get("behavioral_rate",     1.0)), 1.0)
    brate_min = min(float(analysis.get("behavioral_rate_min", brate * 0.6)), 1.0)
    brate_max = min(float(analysis.get("behavioral_rate_max", brate * 1.4)), 1.0)

    est     = demo_base * brate
    est_min = demo_base * brate_min
    est_max = demo_base * brate_max

    # ⑥ 全パネルに対する出現率（%）
    inc     = est     / total * 100
    inc_min = est_min / total * 100
    inc_max = est_max / total * 100

    # ⑦ 必要スクリーニング数
    def req_ss(inc_pct: float) -> int:
        return int(target_n / (inc_pct / 100)) if inc_pct > 0 else 0

    # ⑧ 実現可能性判定
    if est_min >= target_n * 3:
        feasibility, fstatus = "達成見込み十分", "success"
    elif est_min >= target_n:
        feasibility, fstatus = "達成可能（余裕少）", "warning"
    elif est_max >= target_n:
        feasibility, fstatus = "推計次第で達成可能", "warning"
    else:
        feasibility, fstatus = "n数不足リスクあり", "error"

    return {
        "total":              total,
        "age_base":           int(age_base),
        "demo_base":          int(demo_base),
        "gender_ratio":       gender_ratio,
        "pref_ratio":         pref_ratio,
        "include_ages":       include_ages,
        "exclude_ages":       analysis.get("exclude_ages", []),
        "behavioral_rate":    brate,
        "est":                int(est),
        "est_min":            int(est_min),
        "est_max":            int(est_max),
        "inc":                round(inc,     2),
        "inc_min":            round(inc_min, 2),
        "inc_max":            round(inc_max, 2),
        "req_ss":             req_ss(inc),
        "req_ss_conservative": req_ss(inc_min),
        "feasibility":        feasibility,
        "fstatus":            fstatus,
        "target_n":           target_n,
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
            st.write(f"• 性別調整（×{r['gender_ratio']:.2f}）→ {r['demo_base']:,}人")
        if r["pref_ratio"] < 1.0:
            st.write(f"• 地域調整（×{r['pref_ratio']:.2f}）→ {r['demo_base']:,}人")
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
                "例：新NISAなど長期の放置型運用（ほったらかし投資）を行っている方（投資歴3年以内）\n"
                "例：小学生以下の子どもを持つ共働き世帯の母親"
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

        # セッションに保存（ページ再描画時も表示を維持）
        st.session_state["calc_condition"] = condition.strip()
        st.session_state["calc_analysis"]  = analysis
        st.session_state["calc_results"]   = results

        # 履歴に保存
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

    # 結果表示（セッションに保存済みなら再描画後も表示）
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
        st.subheader("CSVで一括入力")
        st.write("以下のフォーマットで作成したCSVをアップロードすると、まとめて登録できます。")
        st.code(
            "属性区分,属性値,人数\n"
            "総数,合計,120000\n"
            "年代,10代,3200\n"
            "年代,20代,18500\n"
            "年代,30代,24100\n"
            "年代,40代,22300\n"
            "年代,50代,19800\n"
            "年代,60代,14200\n"
            "年代,70代以上,8900\n"
            "性別,男性,55000\n"
            "性別,女性,56000\n"
            "都道府県,東京都,18200\n"
            "都道府県,神奈川県,10400",
            language="csv",
        )

        uploaded = st.file_uploader("CSVファイルを選択", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded, header=0)
                current = load_panel()
                count_imported = 0
                for _, row in df.iterrows():
                    cat   = str(row.iloc[0]).strip()
                    val   = str(row.iloc[1]).strip()
                    count = int(str(row.iloc[2]).replace(",", "").strip())
                    if cat == "総数":
                        current["total"] = count
                        count_imported += 1
                    elif cat == "年代" and val in AGE_GROUPS:
                        current["age"][val] = count
                        count_imported += 1
                    elif cat == "性別" and val in GENDERS:
                        current["gender"][val] = count
                        count_imported += 1
                    elif cat == "都道府県":
                        current.setdefault("prefecture", {})[val] = count
                        count_imported += 1
                _save_json(PANEL_FILE, current)
                st.success(f"✅ {count_imported}件を取り込み、保存しました！")
                st.dataframe(df, use_container_width=True, hide_index=True)
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

    # サマリーテーブル
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

    # 詳細表示
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

        # APIキー入力（入力値はconfig.jsonに保存）
        # Streamlit Cloud の Secrets に設定されていれば自動読み込み
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

        # パネルデータの状態を表示
        panel = load_panel()
        if panel["total"] > 0:
            st.success(f"✅ パネル設定済み\n**{panel['total']:,}人**")
        else:
            st.warning("⚠️ パネルデータ未設定\nまずパネルデータを入力してください")

    # ページ切り替え
    panel = load_panel()
    if page == "📊 出現率計算":
        page_calculation(api_key, panel)
    elif page == "⚙️ パネルデータ設定":
        page_panel_setup()
    else:
        page_history()


if __name__ == "__main__":
    main()
