"""
app.py -- the FinSight web app (runs ONLY on your laptop, at localhost).

Run:  streamlit run app.py
Then your browser opens http://localhost:8501

Tabs:
  Overview : money in/out, spending by category / month, top payees, transactions
  Ask      : chat with the AI agent (it sees only your question + column names)
  Import   : upload a PDF (or load the FAKE demo), see the 3 checks, categorise
"""

import os
import sqlite3
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from parse_statement import parse_pdf

REAL_DB = "data/transactions.db"      # your real statement (gitignored)
DEMO_DB = "data/demo.db"              # the fake demo statement, kept separate
DEMO_PDF = "data/fake_statement.pdf"

st.set_page_config(page_title="FinSight", page_icon="◆", layout="wide")

# ---------------------------------------------------------------
# Design tokens (one place for colours)
# ---------------------------------------------------------------
BG, SURFACE, BORDER = "#0B0E14", "#11151E", "#1D2330"
TEXT, MUTED, FAINT = "#E7EAF0", "#8A93A6", "#5B6478"
ACCENT, GOOD, BAD = "#7C9CFF", "#3DD68C", "#F2707A"

# Muted category palette that reads well on dark
CAT_COLORS = {
    "Groceries": "#3DD68C", "Food": "#F5A15B", "Shopping": "#E879A6", "Travel": "#5BB8F5",
    "Fuel": "#E3C85B", "Bills": "#9D8CF2", "Subscriptions": "#B98CF2", "Health": "#F2707A",
    "Entertainment": "#F58FA0", "Education": "#4FD1C5", "Investments": "#7C9CFF",
    "Income": "#3DD68C", "Transfers": "#6B7489", "Cash": "#8A93A6", "Refund": "#56C6D6",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
      font-family: 'Inter', system-ui, sans-serif !important; }}
  .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1240px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  section[data-testid="stSidebar"] {{ background: #0D1118; border-right: 1px solid {BORDER}; }}

  /* top bar */
  .topbar {{ display:flex; align-items:center; justify-content:space-between;
             padding: 6px 0 18px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 8px; }}
  .brand {{ display:flex; align-items:center; gap:12px; }}
  .logo {{ width:34px; height:34px; border-radius:9px; background:{ACCENT}; color:{BG};
           display:flex; align-items:center; justify-content:center; font-weight:700; font-size:18px; }}
  .brand .name {{ font-weight:700; font-size:1.25rem; color:{TEXT}; letter-spacing:-.3px; }}
  .brand .tag {{ color:{MUTED}; font-size:.85rem; margin-top:1px; }}
  .badges span {{ border:1px solid {BORDER}; color:{MUTED}; border-radius:6px; padding:4px 10px;
                  font-size:.78rem; margin-left:8px; background:{SURFACE}; }}
  .badges b {{ color:{GOOD}; font-weight:600; }}

  /* tabs */
  div[data-testid="stTabs"] button p {{ font-size:.95rem; font-weight:500; }}
  div[data-testid="stTabs"] [aria-selected="true"] p {{ color:{TEXT}; }}

  /* metric cards */
  .kpi {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:16px 18px; }}
  .kpi .label {{ color:{MUTED}; font-size:.78rem; font-weight:500; letter-spacing:.2px; }}
  .kpi .value {{ color:{TEXT}; font-size:1.65rem; font-weight:650; margin-top:6px;
                 font-variant-numeric: tabular-nums; letter-spacing:-.5px; }}
  .kpi .sub {{ color:{FAINT}; font-size:.8rem; margin-top:4px; }}
  .kpi .dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px;
               vertical-align:middle; background:var(--c); }}

  /* cards around charts */
  div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{SURFACE}; border-color:{BORDER} !important;
                                                     border-radius:12px; }}
  .h {{ color:{TEXT}; font-weight:600; font-size:.98rem; }}
  .s {{ color:{FAINT}; font-size:.8rem; margin-bottom:6px; }}

  /* check cards */
  .check {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; }}
  .check .t {{ font-weight:600; color:{TEXT}; font-size:.92rem; }}
  .check .d {{ color:{MUTED}; font-size:.84rem; margin-top:4px; font-variant-numeric: tabular-nums; }}
  .check .st {{ float:right; font-size:.75rem; font-weight:600; padding:2px 8px; border-radius:5px; }}
  .pass {{ color:{GOOD}; background:rgba(61,214,140,.1); }}
  .fail {{ color:{BAD}; background:rgba(242,112,122,.12); }}
  .skip {{ color:#E3C85B; background:rgba(227,200,91,.12); }}

  .muted {{ color:{MUTED}; font-size:.86rem; }}
  .stChatMessage {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Chart styling
# ---------------------------------------------------------------
def styled(chart, height):
    """Dark, minimal chart styling applied to every chart."""
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=BORDER, domain=False,
                            tickColor=BORDER, labelFont="Inter", labelFontSize=11.5)
            .configure_legend(labelColor=MUTED, labelFont="Inter", labelFontSize=12, symbolType="circle")
            .configure_text(font="Inter"))


def palette(cats):
    return alt.Scale(domain=list(cats), range=[CAT_COLORS.get(c, "#6B7489") for c in cats])


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def run_checks(df):
    """The parser's 3 independent checks. Returns a list of (name, ok, detail)."""
    df = df.reset_index(drop=True)
    bad = sum(abs(round(df.closing_balance[i - 1] + df.amount[i], 2) - df.closing_balance[i]) > 0.01
              for i in range(1, len(df)))
    checks = [("Balance chain", bad == 0, f"{len(df) - 1 - bad} of {len(df) - 1} rows connect to the previous balance")]
    s = df.attrs.get("summary")
    if s:
        dr, cr = int((df.amount < 0).sum()), int((df.amount > 0).sum())
        checks.append(("Transaction count", (dr, cr) == (s["dr_count"], s["cr_count"]),
                       f"Parsed {dr} debits + {cr} credits · bank: {s['dr_count']} + {s['cr_count']}"))
        if "debits" in s:
            d, c = round(df.withdrawal.sum(), 2), round(df.deposit.sum(), 2)
            checks.append(("Totals", abs(d - s["debits"]) < 0.01 and abs(c - s["credits"]) < 0.01,
                           f"Parsed ₹{d:,.2f} out · ₹{c:,.2f} in · matches bank"
                           if abs(d - s["debits"]) < 0.01 else
                           f"Parsed ₹{d:,.2f} / ₹{c:,.2f} · bank ₹{s['debits']:,.2f} / ₹{s['credits']:,.2f}"))
    else:
        checks.append(("Transaction count", None, "No statement summary in this PDF"))
    return checks


@st.cache_resource(show_spinner="Loading your fine-tuned model (first time ~30 s)...")
def load_hybrid(model_name):
    from hybrid import Hybrid
    return Hybrid(model_name)


def categorise(df, mode, model_name):
    """Add category + category_source. 'Rules only' is instant; 'Hybrid' uses your model."""
    from baseline_rules import learn_rules, predict as rules_predict
    from hybrid import clear_cut_rule
    narr = list(df.narration)
    if mode.startswith("Rules"):
        lookup, keywords = learn_rules(pd.read_csv("data/train.csv"))
        df["category"] = [rules_predict(n, lookup, keywords) for n in narr]
        df["category_source"] = "rules-only"
        return df

    hybrid = load_hybrid(model_name)
    cats, sources = [None] * len(narr), [None] * len(narr)
    todo = []
    for i, n in enumerate(narr):
        c = clear_cut_rule(n)
        if c:
            cats[i], sources[i] = c, "rule"
        else:
            todo.append(i)
    bar = st.progress(0.0, text=f"Categorising {len(todo)} transactions with your model...")
    step = 8
    for k in range(0, len(todo), step):
        chunk = todo[k:k + step]
        preds = hybrid.model.predict([narr[i] for i in chunk])
        for i, p in zip(chunk, preds):
            if p == "INVALID":
                cats[i] = rules_predict(narr[i], hybrid.lookup, hybrid.keywords)
                sources[i] = "fallback"
            else:
                cats[i], sources[i] = p, "model"
        done = min(k + step, len(todo))
        bar.progress(done / max(len(todo), 1), text=f"Categorising... {done}/{len(todo)}")
    bar.empty()
    df["category"], df["category_source"] = cats, sources
    return df


def save(df, db):
    with sqlite3.connect(db) as con:
        df.to_sql("transactions", con, if_exists="replace", index=False)


def load():
    if not Path(DB).exists():
        return None
    with sqlite3.connect(DB) as con:
        df = pd.read_sql("SELECT * FROM transactions", con)
    return df if "category" in df.columns else None


def inr(x):
    """Indian-style grouping: ₹1,23,456"""
    s = str(int(round(abs(x))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1])
        s = f"{head},{tail}"
    return f"₹{s}"


def kpi(col, label, value, sub, color):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="label"><span class="dot"></span>{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def head(text, sub=None):
    st.markdown(f'<div class="h">{text}</div>' + (f'<div class="s">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Data")
    source = st.radio("Data", ["My statement", "Fake demo"], label_visibility="collapsed",
                      help="The demo uses its own database, so it never overwrites your real data.")
    DB = REAL_DB if source == "My statement" else DEMO_DB
    st.markdown("#### Categoriser")
    mode = st.radio("Categoriser", ["Hybrid (fine-tuned model)", "Rules only (instant)"],
                    label_visibility="collapsed",
                    help="Hybrid: clear-cut rules + your fine-tuned model. Accurate, but ~15–25 s per "
                         "transaction on a laptop CPU.")
    model_name = st.selectbox("Model", ["qwen-1.5b", "qwen-0.5b"],
                              help="0.5B is faster on a laptop but less accurate (73% vs 82% on unseen shops).")
    st.markdown("---")
    st.caption("Runs locally. The Ask tab sends only your question and the column names to the "
               "language model, never your transactions.")

data = load()
period = f"{data.date.min()} – {data.date.max()}" if data is not None else "No data loaded"
st.markdown(f"""<div class="topbar">
  <div class="brand"><div class="logo">F</div>
    <div><div class="name">FinSight</div><div class="tag">{source} · {period}</div></div></div>
  <div class="badges"><span><b>●</b> Local</span><span>Qwen 1.5B · LoRA</span><span>Verified vs bank totals</span></div>
</div>""", unsafe_allow_html=True)

tab_dash, tab_ask, tab_stmt = st.tabs(["Overview", "Ask", "Import"])

# ---------------------------------------------------------------
# Tab: Import
# ---------------------------------------------------------------
with tab_stmt:
    left, right = st.columns([3, 2], gap="large")
    with left:
        head("Import a statement", "HDFC PDF → saved as My statement. The fake demo is saved separately.")
        up = st.file_uploader("Statement PDF", type="pdf", label_visibility="collapsed")
        pw = st.text_input("PDF password", type="password", placeholder="Leave empty if the PDF has none")
        c1, c2 = st.columns(2)
        go_upload = c1.button("Import PDF", disabled=up is None, type="primary", width="stretch")
        go_demo = c2.button("Load fake demo statement", width="stretch")
    with right:
        head("Pipeline")
        st.markdown(f"""<div class="muted">
1&nbsp;&nbsp;Parse: word positions → rows (date-anchored)<br>
2&nbsp;&nbsp;Verify: balance chain, count and totals vs the bank<br>
3&nbsp;&nbsp;Categorise: clear-cut rules + fine-tuned model<br>
4&nbsp;&nbsp;Store: local SQLite, read by Overview and Ask</div>""", unsafe_allow_html=True)

    if go_upload or go_demo:
        path, tmp = DEMO_PDF, None
        if go_upload:
            # write the upload to a temp file, read it, delete it straight after
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(up.getvalue())
            tmp.close()
            path = tmp.name
        df = None
        try:
            with st.spinner("Reading the PDF..."):
                df = parse_pdf(path, password=pw or "")
        except ValueError as e:
            st.error(f"Couldn't open the PDF: {e}")
        except SystemExit as e:
            st.error(str(e))
        finally:
            if tmp:
                os.unlink(tmp.name)
        if df is not None:
            st.write("")
            head(f"Read {len(df)} transactions", f"{df.date.min()} – {df.date.max()}")
            cols = st.columns(3)
            for col, (name, ok, detail) in zip(cols, run_checks(df)):
                cls, word = ("pass", "PASS") if ok else (("skip", "N/A") if ok is None else ("fail", "FAIL"))
                col.markdown(f'<div class="check"><span class="st {cls}">{word}</span>'
                             f'<div class="t">{name}</div><div class="d">{detail}</div></div>',
                             unsafe_allow_html=True)
            st.write("")
            df = categorise(df, mode, model_name)
            target = DEMO_DB if go_demo else REAL_DB
            save(df, target)
            st.success(f"Saved as {'Fake demo' if go_demo else 'My statement'}. "
                       "Select it under Data in the sidebar.")

# ---------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------
with tab_dash:
    df = data
    if df is None:
        st.info(f"No {source.lower()} data yet. Use the Import tab.")
    else:
        df["spent"] = (-df.amount).clip(lower=0)
        out, inc = df[df.amount < 0], df[df.amount > 0]
        net = df.amount.sum()
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Money in", inr(inc.amount.sum()), f"{len(inc)} credits", GOOD)
        kpi(k2, "Money out", inr(out.spent.sum()), f"{len(out)} debits", BAD)
        kpi(k3, "Net flow", ("+" if net >= 0 else "−") + inr(net), "in minus out", ACCENT)
        kpi(k4, "Transactions", f"{len(df)}", f"{df.date.str[:7].nunique()} months", MUTED)
        st.write("")

        hide = st.toggle("Exclude transfers to people", value=True,
                         help="Person-to-person transfers usually dwarf everyday spending.")
        view = out[out.category != "Transfers"] if hide else out
        by_cat = view.groupby("category", as_index=False).spent.sum().sort_values("spent", ascending=False)
        by_cat["label"] = by_cat.spent.map(inr)
        by_cat["share"] = by_cat.spent / by_cat.spent.sum()
        order = list(by_cat.category)
        scale = palette(order)

        c1, c2 = st.columns([3, 2], gap="medium")
        with c1, st.container(border=True):
            head("Spending by category", f"{inr(by_cat.spent.sum())} across {len(by_cat)} categories")
            base = alt.Chart(by_cat).encode(y=alt.Y("category:N", sort=order, title=None,
                                                    axis=alt.Axis(labelColor=TEXT, labelFontSize=12.5,
                                                                  labelPadding=10, ticks=False)))
            bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
                x=alt.X("spent:Q", title=None, axis=alt.Axis(format="~s", grid=True, tickCount=5)),
                color=alt.Color("category:N", scale=scale, legend=None),
                tooltip=[alt.Tooltip("category:N", title="Category"),
                         alt.Tooltip("spent:Q", format=",.0f", title="₹"),
                         alt.Tooltip("share:Q", format=".0%", title="Share")])
            text = base.mark_text(align="left", dx=8, fontSize=12, fontWeight=500).encode(
                x="spent:Q", text="label:N", color=alt.value(MUTED))
            st.altair_chart(styled(bars + text, max(240, 32 * len(by_cat))), width="stretch")
        with c2, st.container(border=True):
            top_cat = by_cat.iloc[0] if len(by_cat) else None
            head("Share of spending",
                 f"{top_cat.category} is {top_cat.share:.0%} of the total" if top_cat is not None else "")
            donut = alt.Chart(by_cat).mark_arc(innerRadius=78, outerRadius=118, padAngle=0.015,
                                               cornerRadius=2).encode(
                theta=alt.Theta("spent:Q", stack=True),
                order=alt.Order("spent:Q", sort="descending"),
                color=alt.Color("category:N", scale=scale, sort=order,
                                legend=alt.Legend(title=None, orient="right", labelLimit=110)),
                tooltip=[alt.Tooltip("category:N", title="Category"),
                         alt.Tooltip("share:Q", format=".0%", title="Share")])
            center = alt.Chart(pd.DataFrame({"t": [inr(by_cat.spent.sum())]})).mark_text(
                fontSize=17, fontWeight=600, color=TEXT).encode(text="t:N")
            st.altair_chart(styled(donut + center, 300), width="stretch")

        c3, c4 = st.columns(2, gap="medium")
        with c3, st.container(border=True):
            head("Cumulative spending", "Running total of money out")
            daily = view.groupby("date", as_index=False).spent.sum()
            daily["cumulative"] = daily.spent.cumsum()
            daily["date"] = pd.to_datetime(daily.date)
            area = alt.Chart(daily).mark_area(
                line={"color": ACCENT, "strokeWidth": 2},
                color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                                   stops=[alt.GradientStop(color="rgba(124,156,255,0)", offset=0),
                                          alt.GradientStop(color="rgba(124,156,255,0.28)", offset=1)]),
                interpolate="monotone"
            ).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", grid=False, tickCount=6)),
                     y=alt.Y("cumulative:Q", title=None, axis=alt.Axis(format="~s", tickCount=5)),
                     tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"),
                              alt.Tooltip("cumulative:Q", format=",.0f", title="₹ so far")])
            st.altair_chart(styled(area, 260), width="stretch")
        with c4, st.container(border=True):
            head("By month", "Monthly spending split by category")
            monthly = (view.assign(month=pd.to_datetime(view.date).dt.strftime("%b %Y"), m=view.date.str[:7])
                       .groupby(["m", "month", "category"], as_index=False).spent.sum())
            stack = alt.Chart(monthly).mark_bar(size=46).encode(
                x=alt.X("month:N", title=None, sort=alt.SortField("m"),
                        axis=alt.Axis(labelAngle=0, labelColor=TEXT, ticks=False)),
                y=alt.Y("spent:Q", title=None, axis=alt.Axis(format="~s", tickCount=5)),
                color=alt.Color("category:N", scale=scale, legend=None),
                order=alt.Order("spent:Q", sort="descending"),
                tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("category:N", title="Category"),
                         alt.Tooltip("spent:Q", format=",.0f", title="₹")])
            st.altair_chart(styled(stack, 260), width="stretch")

        with st.container(border=True):
            head("Top payees", "Ranked by total paid")
            top = (view.assign(payee=view.counterparty.replace("", "Unknown"))
                   .groupby(["payee", "category"], as_index=False).spent.agg(["sum", "count"])
                   .sort_values("sum", ascending=False).head(8))
            st.dataframe(top, hide_index=True, width="stretch", column_config={
                "payee": st.column_config.TextColumn("Payee", width="medium"),
                "category": "Category",
                "sum": st.column_config.ProgressColumn("Paid", format="₹%.0f", min_value=0,
                                                       max_value=float(top["sum"].max()) if len(top) else 1),
                "count": st.column_config.NumberColumn("Payments", width="small")})

        with st.container(border=True):
            head("Transactions")
            f1, f2, f3 = st.columns([2, 2, 1])
            q = f1.text_input("Search", placeholder="Search narration, e.g. SWIGGY", label_visibility="collapsed")
            cats = f2.multiselect("Category", sorted(df.category.unique()), placeholder="All categories",
                                  label_visibility="collapsed")
            direction = f3.selectbox("Direction", ["All", "Money out", "Money in"], label_visibility="collapsed")
            show = df.copy()
            if q:
                show = show[show.narration.str.contains(q, case=False, regex=False)]
            if cats:
                show = show[show.category.isin(cats)]
            if direction != "All":
                show = show[show.amount < 0] if direction == "Money out" else show[show.amount > 0]
            st.caption(f"{len(show)} transactions · net {('+' if show.amount.sum() >= 0 else '−')}"
                       f"{inr(show.amount.sum())}")
            st.dataframe(show[["date", "narration", "amount", "category", "category_source"]]
                         .sort_values("date", ascending=False), hide_index=True, width="stretch",
                         height=360, column_config={
                             "date": st.column_config.DateColumn("Date", format="DD MMM"),
                             "narration": st.column_config.TextColumn("Narration", width="large"),
                             "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                             "category": "Category", "category_source": "Source"})

# ---------------------------------------------------------------
# Tab: Ask (the Day 4 agent)
# ---------------------------------------------------------------
with tab_ask:
    from env import load_env
    load_env()
    key = os.environ.get("GROQ_API_KEY", "")
    if data is None:
        st.info(f"No {source.lower()} data yet. Use the Import tab.")
    elif not key.startswith("gsk_"):
        key = st.text_input("Groq API key", type="password",
                            help="Or put GROQ_API_KEY=gsk_... in the .env file so you don't need to paste it.")
    if data is not None and key.startswith("gsk_"):
        @st.cache_resource(show_spinner=False)
        def llm_for(k):
            import baseline_llm
            baseline_llm.MODEL = baseline_llm.pick_model(k)
            return lambda prompt: baseline_llm.call_llm(prompt, k)

        from agent import Agent
        agent = Agent(llm_for(key), db=DB)
        chat = st.session_state.setdefault(f"chat_{source}", [])

        st.markdown(f'<div class="muted">Questions are turned into SQL by the language model, which sees only '
                    f'the question and column names. The query runs here, read-only.</div>',
                    unsafe_allow_html=True)
        st.write("")
        examples = ["How much did we spend on groceries?", "Top 5 payees by amount",
                    "Spending by category in October", "What was the biggest expense?"]
        picked = None
        if not chat:
            cols = st.columns(len(examples))
            for col, ex in zip(cols, examples):
                if col.button(ex, width="stretch"):
                    picked = ex

        for msg in chat:
            with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user"
                                 else ":material/query_stats:"):
                st.markdown(msg["text"])
                if msg.get("note"):
                    st.caption(msg["note"])
                if msg.get("table") is not None and len(msg["table"]) > 1:
                    st.dataframe(msg["table"], hide_index=True)
                if msg.get("sql"):
                    with st.expander("SQL"):
                        st.code(msg["sql"], language="sql")

        q = st.chat_input("Ask about your statement") or picked
        if q:
            chat.append({"role": "user", "text": q})
            with st.spinner("Thinking..."):
                r = agent.ask(q)
            chat.append({"role": "assistant", "text": r["answer"], "note": r.get("note"),
                         "sql": r["sql"], "table": r["table"]})
            st.rerun()
        if chat and st.button("Clear conversation"):
            chat.clear()
            st.rerun()
