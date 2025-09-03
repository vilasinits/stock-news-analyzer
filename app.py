import os, re, time, math, html
import pandas as pd
import numpy as np
import streamlit as st
import feedparser, requests
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus
import yfinance as yf
from yahooquery import search as yq_search
from bs4 import BeautifulSoup

# ---------- Sentiment (FinBERT) ----------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def normalize_date_col(df, col="date"):
    # -> pandas Timestamp at midnight (no timezone)
    s = pd.to_datetime(df[col], utc=True, errors="coerce")
    s = s.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    df[col] = s
    return df

@st.cache_resource
def load_finbert():
    model_name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    return tok, mdl, device

def finbert_score_batch(texts, tok, mdl, device):
    if len(texts) == 0:
        return []
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = mdl(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    # label order: [negative, neutral, positive]
    scores = probs[:, 2] - probs[:, 0]  # pos - neg
    return [{"neg": p[0], "neu": p[1], "pos": p[2], "score": s} for p, s in zip(probs, scores)]

# ---------- Helpers ----------
def clean_html(text):
    return html.unescape(BeautifulSoup(text or "", "lxml").get_text(" ", strip=True))

def parse_time(x):
    if not x: 
        return None
    try:
        # feedparser returns struct_time sometimes
        if isinstance(x, time.struct_time):
            return datetime.fromtimestamp(time.mktime(x), tz=timezone.utc)
        # strings fallback
        return pd.to_datetime(x, utc=True)
    except Exception:
        return None

def uniq(seq):
    seen = set()
    out = []
    for x in seq:
        k = (x.get("title",""), x.get("link",""))
        if k in seen: 
            continue
        seen.add(k)
        out.append(x)
    return out

# ---------- Source builders ----------
def google_news_rss(query, lang="en", country="US"):
    # Google News RSS is reliable and broad; we can filter sites via query operators
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={country}:{lang}"

RSS_SOURCES = {
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "CNN Business": "http://rss.cnn.com/rss/money_latest.rss",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Bloomberg (Top)": "https://feeds.simplecast.com/54nAGcIl",  # podcasts; headlines via Google News preferred
    "Times Now (Business)": "https://www.timesnownews.com/rssfeeds/65887841.cms",  # India business feed
    # Fallback to Google News queries for specific outlets:
    # Example: site:bloomberg.com CompanyName
}

def fetch_rss(url, limit=50):
    try:
        feed = feedparser.parse(url)
        items = []
        for e in (feed.entries or [])[:limit]:
            items.append({
                "title": clean_html(getattr(e, "title", "")),
                "summary": clean_html(getattr(e, "summary", "")),
                "published": parse_time(getattr(e, "published_parsed", None) or getattr(e, "published", None)),
                "link": getattr(e, "link", ""),
                "source": feed.feed.get("title", url),
            })
        return items
    except Exception:
        return []

def news_for_company(name, max_items=120, days_back=7):
    # 1) Broad Google News
    items = fetch_rss(google_news_rss(name), limit=max_items)

    # 2) Site-filtered Google News for key outlets
    for site in ["bbc.com", "cnn.com", "bloomberg.com", "reuters.com", "ft.com", "wsj.com", "theguardian.com"]:
        items += fetch_rss(google_news_rss(f'{name} site:{site}'), limit=60)

    # 3) General business feeds (catch-all)
    for lbl, url in RSS_SOURCES.items():
        items += fetch_rss(url, limit=40)

    items = uniq(items)
    # Filter by recency and by explicit name mention to keep it relevant
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    def relevant(it):
        text = f"{it.get('title','')} {it.get('summary','')}".lower()
        return (it.get("published") and it["published"] >= cutoff) and (name.lower() in text)
    filt = list(filter(relevant, items))
    return sorted(filt, key=lambda x: x["published"] or datetime(1970,1,1,tzinfo=timezone.utc), reverse=True)

# ---------- Ticker resolution ----------
def resolve_ticker(company_name):
    # Use Yahoo Query search to find best matching equity
    try:
        res = yq_search(company_name)
        quotes = res.get("quotes", [])
        quotes = [q for q in quotes if q.get("quoteType") == "EQUITY" and q.get("symbol")]
        if not quotes:
            return None, None
        best = quotes[0]  # could add fuzzy ranking here
        return best.get("symbol"), best.get("longname") or best.get("shortname") or best.get("symbol")
    except Exception:
        return None, None

# ---------- Feature engineering ----------
def aggregate_sentiment(rows):
    if len(rows) == 0:
        return pd.DataFrame(columns=["date","mean_score","pos_share","neg_share","n"])
    df = pd.DataFrame(rows)
    # df["date"] = pd.to_datetime(df["published"]).dt.tz_convert("UTC").dt.date
    df["date"] = pd.to_datetime(df["published"], utc=True).dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)

    # daily aggregation
    out = df.groupby("date").apply(lambda g: pd.Series({
        "mean_score": g["score"].mean(),
        "pos_share": (g["pos"] > 0.7).mean(),
        "neg_share": (g["neg"] > 0.7).mean(),
        "n": len(g)
    })).reset_index()
    return out

def price_frame(ticker, lookback_days=365):
    from datetime import datetime, timedelta
    import pandas as pd
    import yfinance as yf

    end = datetime.utcnow().date() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    # auto_adjust=True gives a split/dividend adjusted Close -> simpler & reliable
    px = yf.download(
        ticker, start=str(start), end=str(end),
        auto_adjust=True, progress=False, group_by="column"
    )
    if px is None or px.empty:
        return pd.DataFrame()

    # Handle both single-index and MultiIndex column layouts
    if isinstance(px.columns, pd.MultiIndex):
        # Try to get 'Close' for this ticker
        if ('Close', ticker) in px.columns:
            s = px[('Close', ticker)]
        elif ('Adj Close', ticker) in px.columns:
            s = px[('Adj Close', ticker)]
        else:
            # Fallback: take the 'Close' slice (or the first column)
            try:
                s = px.xs('Close', level=0, axis=1).squeeze()
            except Exception:
                s = px.iloc[:, 0].squeeze()
    else:
        if 'Close' in px.columns:
            s = px['Close']
        elif 'Adj Close' in px.columns:
            s = px['Adj Close']
        else:
            # Fallback to first numeric column
            s = px.select_dtypes('number').iloc[:, 0]

    df = s.to_frame('close').copy()
    df['ret1'] = df['close'].pct_change()
    df['date'] = pd.to_datetime(df.index).date
    return df.reset_index(drop=True)


def build_training_table(sent_daily, px):
    if sent_daily.empty or px.empty:
        return pd.DataFrame()
    df = pd.merge(px[["date","close","ret1"]], sent_daily, on="date", how="left").sort_values("date")
    df = pd.merge(px[["date","close","ret1"]], sent_daily, on="date", how="left").sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)  # ensure datetime64[ns]

    # fill sentiment gaps with 0 (no news) and 0 counts
    for c, val in [("mean_score", 0.0), ("pos_share", 0.0), ("neg_share", 0.0), ("n", 0.0)]:
        df[c] = df[c].fillna(val)
    # rolling features
    df["sent_roll3"] = df["mean_score"].rolling(3).mean()
    df["sent_roll7"] = df["mean_score"].rolling(7).mean()
    df["news_vol7"] = df["n"].rolling(7).mean()
    df["mom_5"] = df["close"].pct_change(5)
    df["vol_5"] = df["ret1"].rolling(5).std()
    # target: next-day return
    df["y_tplus1"] = df["ret1"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

# ---------- Model ----------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def train_and_predict_next(df):
    feats = ["mean_score","pos_share","neg_share","n","sent_roll3","sent_roll7","news_vol7","mom_5","vol_5"]
    if df.empty or len(df) < 60:
        return None, None, None
    X = df[feats].values
    y = df["y_tplus1"].values
    # simple expanding window: train on all but last, predict last
    X_train, y_train = X[:-1], y[:-1]
    X_last = X[-1:].copy()
    mdl = LinearRegression().fit(X_train, y_train)
    pred = float(mdl.predict(X_last)[0])
    r2 = r2_score(y_train, mdl.predict(X_train))
    return mdl, pred, r2

# ---------- OOS preds, bands, and simple 7-day simulation ----------

from numpy.linalg import inv

FEATS = ["mean_score","pos_share","neg_share","n","sent_roll3","sent_roll7","news_vol7","mom_5","vol_5"]

def walkforward_oos(df, feats=FEATS, min_train=60):
    """Expanding-window one-step-ahead predictions aligned to the date they predict."""
    dates, yhat = [], []
    for i in range(min_train, len(df)-1):
        X_tr = df[feats].iloc[:i].values
        y_tr = df["y_tplus1"].iloc[:i].values
        X_pred = df[feats].iloc[i:i+1].values  # predicts return on next date
        mdl = LinearRegression().fit(X_tr, y_tr)
        yhat.append(float(mdl.predict(X_pred)[0]))
        dates.append(pd.to_datetime(df["date"].iloc[i+1]))
    out = pd.DataFrame({"date": pd.to_datetime(dates), "yhat": yhat})
    actual = df[["date","ret1"]].iloc[1:].rename(columns={"ret1":"actual_ret"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    actual["date"] = pd.to_datetime(actual["date"]).dt.tz_localize(None)

    out = out.merge(actual, on="date", how="left")
    out["resid"] = out["actual_ret"] - out["yhat"]
    return out

def sentiment_price_last_k(px, oos_df, k=7, sigma=None):
    """Price implied by sentiment (prev close * (1+yhat)) for the last k predicted days."""
    pxd = px[["date","close"]].copy()
    pxd["prev_close"] = pxd["close"].shift(1)
    z = oos_df.merge(pxd[["date","prev_close"]], on="date", how="left").dropna()
    z["sentiment_price"] = z["prev_close"] * (1.0 + z["yhat"])
    if sigma is not None:
        # approx 90% bands on price via ±1.64σ on return
        z["lower"] = z["prev_close"] * (1.0 + z["yhat"] - 1.64*sigma)
        z["upper"] = z["prev_close"] * (1.0 + z["yhat"] + 1.64*sigma)
    return z.tail(k)[["date","sentiment_price","yhat","lower","upper"] if sigma is not None else ["date","sentiment_price","yhat"]]

def forecast_next_k(train_df, px, k=7, feats=FEATS, sentiment_mode="zero", sigma=None, sims=800, seed=0):
    """Forecast next k days of price by iterating the linear model.
       sentiment_mode: 'zero' (no future news) or 'persist' (carry 7d means)."""
    mdl = LinearRegression().fit(train_df[feats].values, train_df["y_tplus1"].values)

    last_date = pd.to_datetime(train_df["date"].iloc[-1])
    last_close = float(px["close"].iloc[-1])

    # make rolling series we can extend
    close_ser = px["close"].copy()
    ret_ser   = px["ret1"].copy()
    s_mean = train_df["mean_score"].copy()
    s_pos  = train_df["pos_share"].copy()
    s_neg  = train_df["neg_share"].copy()
    s_n    = train_df["n"].copy()

    preds, dates = [], []
    for step in range(1, k+1):
        if sentiment_mode == "zero":
            mean_score = pos_share = neg_share = n = 0.0
        elif sentiment_mode == "persist":
            mean_score = float(s_mean.iloc[-7:].mean())
            pos_share  = float(s_pos.iloc[-7:].mean())
            neg_share  = float(s_neg.iloc[-7:].mean())
            n          = float(s_n.iloc[-7:].mean())
        elif sentiment_mode == "decay":
            decay = 0.8 ** step   # fade 20% per day ahead
            mean_score = float(s_mean.iloc[-7:].mean()) * decay
            pos_share  = float(s_pos.iloc[-7:].mean()) * decay
            neg_share  = float(s_neg.iloc[-7:].mean()) * decay
            n          = float(s_n.iloc[-7:].mean()) * decay

        mom_5 = (close_ser.iloc[-1]/close_ser.iloc[-5]-1) if len(close_ser) >= 5 else 0.0
        vol_5 = float(ret_ser.iloc[-5:].std()) if len(ret_ser) >= 5 else float(ret_ser.std())

        sent_roll3 = float(pd.Series(list(s_mean.iloc[-2:]) + [mean_score]).rolling(3).mean().iloc[-1]) if len(s_mean)>=2 else mean_score
        sent_roll7 = float(pd.Series(list(s_mean.iloc[-6:]) + [mean_score]).rolling(7).mean().iloc[-1]) if len(s_mean)>=6 else mean_score
        news_vol7  = float(pd.Series(list(s_n.iloc[-6:]) + [n]).rolling(7).mean().iloc[-1]) if len(s_n)>=6 else n

        X = np.array([[mean_score,pos_share,neg_share,n,sent_roll3,sent_roll7,news_vol7,mom_5,vol_5]])
        rhat = float(mdl.predict(X)[0])
        # Clamp daily return to realistic band
        # rhat = float(mdl.predict(X)[0])
        rhat = np.clip(rhat, -0.03, 0.03)   # limit ±3% per day
        # preds.append(rhat)

        preds.append(rhat)
        dates.append(last_date + pd.Timedelta(days=step))

        # update series
        new_close = close_ser.iloc[-1]*(1.0 + rhat)
        close_ser = pd.concat([close_ser, pd.Series([new_close])], ignore_index=True)
        ret_ser   = pd.concat([ret_ser,   pd.Series([rhat])],      ignore_index=True)
        s_mean = pd.concat([s_mean, pd.Series([mean_score])], ignore_index=True)
        s_pos  = pd.concat([s_pos,  pd.Series([pos_share])],  ignore_index=True)
        s_neg  = pd.concat([s_neg,  pd.Series([neg_share])],  ignore_index=True)
        s_n    = pd.concat([s_n,    pd.Series([n])],          ignore_index=True)

    # deterministic path
    prices = []
    cp = last_close
    for r in preds:
        cp *= (1.0 + r)
        prices.append(cp)

    # MC bands
    lower = upper = None
    if sigma is not None and sims > 0:
        rng = np.random.default_rng(seed)
        mc = np.empty((sims, k))
        for s in range(sims):
            cp = last_close
            for j in range(k):
                rr = rng.normal(preds[j], sigma)
                cp *= (1.0 + rr)
                mc[s, j] = cp
        lower = np.percentile(mc, 5, axis=0)
        upper = np.percentile(mc, 95, axis=0)

    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "pred_ret": preds,
        "pred_price": prices,
        "lower": lower, "upper": upper
    })

def coef_with_se(df, feats=FEATS):
    """Linear regression coefficients with standard errors (for bars w/ error bars)."""
    X = df[feats].values
    y = df["y_tplus1"].values
    mdl = LinearRegression().fit(X, y)
    yhat = mdl.predict(X)
    resid = y - yhat
    n, p = X.shape
    sigma2 = (resid @ resid) / (n - p)
    XtX_inv = inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    return pd.DataFrame({"feature": feats, "coef": mdl.coef_, "se": se}).sort_values("coef")

# ---------- UI ----------
st.set_page_config(page_title="Stock News Sentiment & Next-Day Predictor", layout="wide")
st.title("📰→📈 Stock Sentiment & Price Predictor")
st.caption("Type a company name. App fetches multi-source news, labels sentiment with FinBERT, joins with prices, and predicts next-day return.")

company = st.text_input("Company name", value="Apple")
days_back = st.slider("News window (days)", 3, 30, 7)
lookback_days = st.slider("Price lookback (days)", 120, 730, 365)
# 7-day forecast (choose future sentiment assumption in the UI)
assump = st.selectbox(
    "Future sentiment assumption for forecast",
    [
        "zero (no future news)",
        "persist (carry last 7d averages)",
        "decay (fade last 7d averages → neutral)"
    ]
)

if assump.startswith("zero"):
    assump_key = "zero"
elif assump.startswith("persist"):
    assump_key = "persist"
else:
    assump_key = "decay"

sims = st.slider("Monte Carlo paths for bands", 200, 2000, 800, 100)


if st.button("Run"):
    with st.spinner("Resolving ticker…"):
        ticker, resolved_name = resolve_ticker(company)
    if not ticker:
        st.error("Could not resolve a ticker. Try a different name or add the ticker directly (e.g., 'Apple AAPL').")
        st.stop()

    st.success(f"Resolved: **{resolved_name}** → **{ticker}**")

    with st.spinner("Fetching news…"):
        news_items = news_for_company(company, days_back=days_back)
    st.write(f"Fetched {len(news_items)} news items in the last {days_back} days.")

    titles = [it["title"] for it in news_items]
    summaries = [it["summary"] for it in news_items]
    texts = [ (t or "") + ". " + (s or "") for t, s in zip(titles, summaries) ]

    if len(texts) == 0:
        st.warning("No recent news matched this company; prediction will rely on price-only features.")
        sent_labeled = []
    else:
        tok, mdl, device = load_finbert()
        batch = 24
        sent_labeled = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            scores = finbert_score_batch(chunk, tok, mdl, device)
            for sc, it in zip(scores, news_items[i:i+batch]):
                row = dict(it)
                row.update(sc)
                sent_labeled.append(row)

    # Show recent labeled headlines
    if sent_labeled:
        show_df = pd.DataFrame(sent_labeled)[["published","source","title","score","pos","neg","link"]].copy()
        show_df["published"] = pd.to_datetime(show_df["published"]).dt.tz_convert("UTC")
        st.subheader("Latest headlines & sentiment")
        st.dataframe(show_df.sort_values("published", ascending=False), use_container_width=True)

    with st.spinner("Fetching prices…"):
        px = price_frame(ticker, lookback_days=lookback_days)
    if px.empty:
        st.error("Could not fetch price history.")
        st.stop()

    sent_daily = aggregate_sentiment(sent_labeled) if sent_labeled else pd.DataFrame(columns=["date","mean_score","pos_share","neg_share","n"])
    # px = price_frame(ticker, lookback_days=lookback_days)
    # sent_daily = aggregate_sentiment(sent_labeled) if sent_labeled else pd.DataFrame(columns=["date","mean_score","pos_share","neg_share","n"])
    if not px.empty and "date" in px:
        px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None).dt.normalize()

    if not sent_daily.empty and "date" in sent_daily:
        sent_daily["date"] = pd.to_datetime(sent_daily["date"]).dt.tz_localize(None).dt.normalize()

    train_df = build_training_table(sent_daily, px)

    if train_df.empty:
        st.error("Not enough data to train. Try expanding lookback or news window.")
        st.stop()

    mdl, pred, r2 = train_and_predict_next(train_df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Resolved Ticker", ticker)
    with col2:
        st.metric("Training R² (in-sample)", f"{(r2 if r2 is not None else 0):.3f}")
    with col3:
        st.metric("Predicted next-day return", f"{(pred or 0)*100:.2f}%")

 
    # ---------- Plots & Diagnostics ----------

    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as pex

    # Out-of-sample predictions across history
    oos = walkforward_oos(train_df, FEATS)
    sigma = float(oos["resid"].dropna().std()) if len(oos) else 0.0   # std of return residuals

    # Last-7d sentiment-implied prices (with error bars)
    sentiment_last7 = sentiment_price_last_k(px, oos, k=7, sigma=(sigma if sigma>0 else None))

    
    forecast_df = forecast_next_k(train_df, px, k=7, sentiment_mode=assump_key, sigma=(sigma if sigma>0 else None), sims=sims)

    def plot_main_plotly(px, last7_df, fc_df, assump_key):
        fig = go.Figure()

        # Actual historical price
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(px["date"]), y=px["close"],
            mode="lines", name="Actual Price",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>"
        ))

        # Last 7 days sentiment-implied (with error bars if present)
        if last7_df is not None and not last7_df.empty:
            yy = last7_df["sentiment_price"]
            err_plus  = (last7_df["upper"] - yy) if "upper" in last7_df else None
            err_minus = (yy - last7_df["lower"]) if "lower" in last7_df else None
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(last7_df["date"]), y=yy,
                mode="markers", name="Sentiment-based Price (last 7d)",
                error_y=dict(
                    type="data", symmetric=False,
                    array=None if err_plus is None else err_plus,
                    arrayminus=None if err_minus is None else err_minus,
                    visible=False if err_plus is None else True
                ),
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Sentiment px=%{y:.2f}<extra></extra>"
            ))

        # Next 7 days forecast + band
        if fc_df is not None and not fc_df.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(fc_df["date"]), y=fc_df["pred_price"],
                mode="lines+markers", line=dict(dash="dash"),
                name=f"Predicted Price (next 7d) [{assump_key}]",
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Pred px=%{y:.2f}<extra></extra>"
            ))
            if "lower" in fc_df and fc_df["lower"].notna().any():
                # Upper trace first, then lower with fill to create the band
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(fc_df["date"]), y=fc_df["upper"],
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(fc_df["date"]), y=fc_df["lower"],
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(255, 0, 0, 0.2)",
                    name="Forecast 90% band", hoverinfo="skip"
                ))

        fig.update_layout(
            title="Sentiment vs Price — last 7d & next 7d",
            yaxis_title="Price",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig

    # In the UI:
    st.subheader("Sentiment vs Price (interactive)")
    st.plotly_chart(
        plot_main_plotly(px, sentiment_last7, forecast_df, assump_key),
        use_container_width=True, theme="streamlit"
    )

    # Diagnostics: residuals time series (walk-forward)
    st.subheader("Residuals (walk-forward, actual − predicted returns)")
    fig_r, ax_r = plt.subplots(figsize=(10,3))
    ax_r.plot(oos["date"], oos["resid"], marker=".", linestyle="none")
    ax_r.axhline(0, linestyle="--", linewidth=1)
    ax_r.set_ylabel("Residual")
    fig_r.tight_layout()
    st.pyplot(fig_r)

    # Calibration: predicted vs actual returns (walk-forward)
    st.subheader("Calibration: predicted vs actual next-day returns")
    fig_c, ax_c = plt.subplots(figsize=(5,5))
    ax_c.scatter(oos["yhat"], oos["actual_ret"], s=10, alpha=0.6)
    lims = [min(ax_c.get_xlim()[0], ax_c.get_ylim()[0]), max(ax_c.get_xlim()[1], ax_c.get_ylim()[1])]
    ax_c.plot(lims, lims, "--", linewidth=1)
    ax_c.set_xlabel("Predicted return")
    ax_c.set_ylabel("Actual return")
    fig_c.tight_layout()
    st.pyplot(fig_c)

    # Rolling correlation between sentiment and next-day returns
    st.subheader("Rolling correlation: daily sentiment vs next-day return")
    roll = (train_df[["date","mean_score"]]
        .merge(train_df[["date","ret1"]].rename(columns={"ret1":"next_ret"}).shift(-1), on="date"))
    roll = roll.dropna()
    roll["dt"] = pd.to_datetime(roll["date"]).dt.tz_localize(None)

    # roll["dt"] = pd.to_datetime(roll["date"])
    roll["corr_30"] = roll["mean_score"].rolling(30).corr(roll["next_ret"])
    fig_corr, ax_corr = plt.subplots(figsize=(10,3))
    ax_corr.plot(roll["dt"], roll["corr_30"])
    ax_corr.axhline(0, linestyle="--", linewidth=1)
    ax_corr.set_ylabel("ρ (30d)")
    fig_corr.tight_layout()
    st.pyplot(fig_corr)

    # Quick trust metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    oos_clean = oos.dropna()
    oos_r2 = r2_score(oos_clean["actual_ret"], oos_clean["yhat"]) if len(oos_clean)>5 else float("nan")
    oos_mae = mean_absolute_error(oos_clean["actual_ret"], oos_clean["yhat"]) if len(oos_clean)>5 else float("nan")
    coverage = float(((oos_clean["actual_ret"] >= oos_clean["yhat"] - 1.64*sigma) &
                    (oos_clean["actual_ret"] <= oos_clean["yhat"] + 1.64*sigma)).mean()) if sigma>0 else float("nan")
    st.caption(f"Walk-forward R²: {oos_r2:.3f} · MAE: {oos_mae:.4f} · 90% band coverage: {coverage:.2%}")
