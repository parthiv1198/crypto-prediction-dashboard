import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

# --- 1. CONFIGURATION AND MODEL DEFINITIONS ---
st.set_page_config(layout="wide", page_title="Crypto Prediction Engine")

def load_css():
    st.markdown("""
    <style>
        /* --- Robust Page Transition Animation --- */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .main .block-container > div:nth-child(1) {
            animation: fadeIn 0.6s ease-in-out;
        }

        /* Animated Gradient for the Main Title */
        @keyframes gradient { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }
        h1 {
            background: linear-gradient(to right, #636E72, #B2BEC3, #636E72);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            animation: gradient 10s ease infinite; background-size: 200% 200%;
        }

        /* Staggered Animation for Metric Cards */
        @keyframes slideUpFadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        [data-testid="stMetric"] { animation: slideUpFadeIn 0.5s ease-out forwards; opacity: 0; }
        div:has(> [data-testid="stMetric"]):nth-child(1) [data-testid="stMetric"] { animation-delay: 0.1s; }
        div:has(> [data-testid="stMetric"]):nth-child(2) [data-testid="stMetric"] { animation-delay: 0.2s; }
        div:has(> [data-testid="stMetric"]):nth-child(3) [data-testid="stMetric"] { animation-delay: 0.3s; }
        div:has(> [data-testid="stMetric"]):nth-child(4) [data-testid="stMetric"] { animation-delay: 0.4s; }

        /* Subtle Button Hover Effect */
        .stButton > button { transition: all 0.3s ease-in-out !important; }
        .stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0 0 12px rgba(230, 230, 230, 0.2);
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
    </style>
    """, unsafe_allow_html=True)

ASSET_CONFIG = {
    "Bitcoin (BTC)": {
        "model_path": "bitcoin_hybrid_pytorch_model.pth", "npz_path": "bitcoin_hybrid_pytorch_data.npz", "raw_data_path": "output_10_rows.csv",
        "time_col": "Timestamp", "timeframe_type": "minute", "horizons": {
            "5 Minutes": {"h": 5, "t": "Target_5m_Pct_Change"}, "30 Minutes": {"h": 30, "t": "Target_30m_Pct_Change"},
            "4 Hours": {"h": 240, "t": "Target_4h_Pct_Change"}, "24 Hours": {"h": 1440, "t": "Target_24h_Pct_Change"},}},
    "Ethereum (ETH)": {
        "model_path": "ethereum_hybrid_pytorch_model.pth", "npz_path": "ethereum_hybrid_pytorch_data.npz", "raw_data_path": "ETH_1min.csv",
        "time_col": "Unix Timestamp", "timeframe_type": "minute", "horizons": {
            "5 Minutes": {"h": 5, "t": "Target_5m_Pct_Change"}, "30 Minutes": {"h": 30, "t": "Target_30m_Pct_Change"},
            "4 Hours": {"h": 240, "t": "Target_4h_Pct_Change"}, "24 Hours": {"h": 1440, "t": "Target_24h_Pct_Change"},}},
    "XPR": {
        "model_path": "xpr_hybrid_pytorch_model.pth", "npz_path": "xpr_hybrid_pytorch_data.npz", "raw_data_path": "XPR_Daily.csv",
        "time_col": "Open time", "timeframe_type": "minute", "horizons": {
            "5 Minutes": {"h": 5, "t": "Target_5m_Pct_Change"}, "30 Minutes": {"h": 30, "t": "Target_30m_Pct_Change"},
            "4 Hours": {"h": 240, "t": "Target_4h_Pct_Change"}, "24 Hours": {"h": 1440, "t": "Target_24h_Pct_Change"},}},
    "Solana (SOL)": {
        "model_path": "solana_hybrid_pytorch_model.pth", "npz_path": "solana_hybrid_pytorch_data.npz", "raw_data_path": "Solana_daily.csv",
        "time_col": "time", "timeframe_type": "day", "horizons": {
            "1 Day": {"h": 1, "t": "Target_1_Day_Pct_Change"}, "7 Days": {"h": 7, "t": "Target_7_Day_Pct_Change"},
            "30 Days": {"h": 30, "t": "Target_30_Day_Pct_Change"}, "90 Days": {"h": 90, "t": "Target_90_Day_Pct_Change"},}},
    "Dogecoin (DOGE)": {
        "model_path": "doge_hybrid_pytorch_model.pth", "npz_path": "doge_hybrid_pytorch_data.npz", "raw_data_path": "DOGE-USD.csv",
        "time_col": "Date", "timeframe_type": "day", "horizons": {
            "1 Day": {"h": 1, "t": "Target_1_Day_Pct_Change"}, "7 Days": {"h": 7, "t": "Target_7_Day_Pct_Change"},
            "30 Days": {"h": 30, "t": "Target_30_Day_Pct_Change"}, "90 Days": {"h": 90, "t": "Target_90_Day_Pct_Change"},}}
}

class PyTorchHybridLSTM(nn.Module):
    def __init__(self, i, h, n): super(PyTorchHybridLSTM, self).__init__(); self.lstm1=nn.LSTM(i,h,batch_first=True); self.dropout1=nn.Dropout(0.2); self.lstm2=nn.LSTM(h,h,batch_first=True); self.dropout2=nn.Dropout(0.2); self.fc1=nn.Linear(h,25); self.relu=nn.ReLU(); self.fc2=nn.Linear(25,n)
    def forward(self, x): o,_=self.lstm1(x); o=self.dropout1(o); o,_=self.lstm2(o); o=self.dropout2(o); o=o[:,-1,:]; o=self.fc1(o); o=self.relu(o); o=self.fc2(o); return o

# --- 2. BACKTESTING ENGINE & HELPERS ---
@st.cache_resource
def load_model(asset_name):
    config = ASSET_CONFIG[asset_name]; device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(config['npz_path'], allow_pickle=True) as data: X_test, target_cols = data['X_test'], data['target_cols']
    model = PyTorchHybridLSTM(X_test.shape[2], 40, len(target_cols)).to(device); model.load_state_dict(torch.load(config['model_path'], map_location=torch.device(device))); model.eval()
    return model

@st.cache_data
def get_predictions(_model, asset_name):
    config = ASSET_CONFIG[asset_name]; device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(config['npz_path'], allow_pickle=True) as data: X_test, target_cols = data['X_test'], data['target_cols']
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test.astype(np.float32))), batch_size=512)
    all_preds = []
    with torch.no_grad():
        for b in test_loader: p = _model(b[0].to(device)); all_preds.append(p.cpu().numpy())
    return np.concatenate(all_preds, axis=0), target_cols

def run_backtest(asset_name, predictions, target_cols, strategy_config):
    config = ASSET_CONFIG[asset_name]; raw_df = pd.read_csv(config['raw_data_path'])
    if raw_df[config['time_col']].dtype == 'object': raw_df['Timestamp'] = pd.to_datetime(raw_df[config['time_col']])
    elif raw_df[config['time_col']].iloc[0] > 10**12: raw_df['Timestamp'] = pd.to_datetime(raw_df[config['time_col']] / 1000, unit='s')
    else: raw_df['Timestamp'] = pd.to_datetime(raw_df[config['time_col']], unit='s')
    raw_df.sort_values('Timestamp', inplace=True); raw_df.reset_index(drop=True, inplace=True)
    with np.load(config['npz_path'], allow_pickle=True) as data: X_test = data['X_test']
    required_len = len(X_test) + strategy_config['holding_period']; test_prices_df = raw_df.tail(required_len).copy().reset_index(drop=True)
    target_col_index = np.where(target_cols == strategy_config['prediction_target'])[0][0]; final_predictions = predictions[:, target_col_index]
    trades = []
    if strategy_config['type'] == 'Long-Only':
        for i in range(len(final_predictions)):
            if i + strategy_config['holding_period'] >= len(test_prices_df): break
            if final_predictions[i] > strategy_config['trade_threshold']: entry = test_prices_df['Close'].iloc[i]; exit_price = test_prices_df['Close'].iloc[i + strategy_config['holding_period']]; trades.append({"return_pct": ((exit_price - entry) / entry) * 100})
    elif strategy_config['type'] == 'Long/Short':
        for i in range(len(final_predictions)):
            if i + strategy_config['holding_period'] >= len(test_prices_df): break
            prediction = final_predictions[i]; entry_price = test_prices_df['Close'].iloc[i]; exit_price = test_prices_df['Close'].iloc[i + strategy_config['holding_period']]
            if prediction > strategy_config['trade_threshold']: trades.append({"return_pct": ((exit_price - entry_price) / entry_price) * 100})
            elif prediction < -strategy_config['trade_threshold']: trades.append({"return_pct": ((entry_price - exit_price) / entry_price) * 100})
    if not trades: return None, None, None, None
    trades_df = pd.DataFrame(trades); total_trades = len(trades_df); win_df = trades_df[trades_df['return_pct'] > 0]; win_rate = len(win_df) / total_trades * 100 if total_trades > 0 else 0
    initial_capital = 10000; returns_series = trades_df['return_pct'] / 100; pnl_series = returns_series * initial_capital
    equity_curve = pnl_series.cumsum() + initial_capital; final_capital = equity_curve.iloc[-1]; strategy_total_return = (final_capital - initial_capital) / initial_capital * 100
    days_in_test = (test_prices_df['Timestamp'].iloc[-1] - test_prices_df['Timestamp'].iloc[0]).days if not test_prices_df.empty else 1
    if returns_series.std() != 0 and days_in_test > 0 and total_trades > 0:
        sharpe_ratio = returns_series.mean() / returns_series.std()
        if config['timeframe_type'] == 'minute': ann_factor = np.sqrt(365*24*60 / (days_in_test*24*60/total_trades)) if (days_in_test*24*60/total_trades) > 0 else 1
        else: ann_factor = np.sqrt(365 / (days_in_test/total_trades)) if (days_in_test/total_trades) > 0 else 1
        annualized_sharpe = sharpe_ratio * ann_factor
    else: annualized_sharpe = 0
    peak = equity_curve.expanding(min_periods=1).max(); drawdown = (equity_curve - peak) / peak; max_dd = drawdown.min() * -100 if not drawdown.empty else 0
    years_in_test = days_in_test/365.25 if days_in_test > 0 else 1; ann_return = ((final_capital/initial_capital)**(1/years_in_test)-1)*100 if years_in_test > 0 else strategy_total_return
    calmar = ann_return/max_dd if max_dd > 0 else 0
    buy_hold = (test_prices_df['Close'].iloc[-1] - test_prices_df['Close'].iloc[0]) / test_prices_df['Close'].iloc[0] * 100
    results = {"Strategy Total Return (%)": strategy_total_return, "Buy & Hold Return (%)": buy_hold, "Annualized Sharpe Ratio": annualized_sharpe, "Calmar Ratio": calmar, "Max Drawdown (%)": max_dd, "Win Rate (%)": win_rate, "Total Trades": total_trades}
    return results, equity_curve, peak, drawdown

# --- 3. STREAMLIT UI AND APPLICATION LOGIC ---

def render_landing_page():
    st.title("Welcome to the Crypto Price Prediction Engine")
    st.subheader("A Dashboard for Backtesting ML-Driven Trading Strategies")
    st.markdown("---")
    st.markdown("""This application is the culmination of a comprehensive project to build and validate a predictive modeling pipeline for cryptocurrencies. Here, you can use our best model—the **Hybrid LSTM**—to simulate trading strategies across multiple assets and time horizons.""")
    st.info("**Disclaimer:** This is a research tool for educational and analytical purposes only. The results are based on historical data and do not constitute financial advice. Past performance is not indicative of future results.")
    
    # --- RESTORED: DETAILED LANDING PAGE TEXT ---
    st.header("How to Use This Dashboard")
    st.markdown("""
    1.  **Navigate to the Strategy Dashboard:** Use the sidebar on the left.
    2.  **Select an Asset:** Choose from Bitcoin, Ethereum, Solana, and more.
    3.  **Choose a Time Horizon:** Select the prediction timeframe you want to test (e.g., 24 hours for BTC, or 7 days for SOL).
    4.  **Pick a Strategy Type:**
        -   **Long-Only:** The strategy will only buy when a significant price increase is predicted.
        -   **Long/Short:** The strategy will buy on positive predictions and sell (short) on negative predictions.
    5.  **Adjust the Confidence Threshold:** The "Base Threshold" slider controls the strategy's sensitivity. A higher threshold means the model must be more "confident" in its prediction before a trade is triggered. This usually results in fewer, but higher-quality, trades.
    6.  **Run the Backtest:** Click the button to run the simulation. The results will be displayed and will remain on the page until you run a new backtest.
    """)
    st.header("How to Interpret the Results")
    st.markdown("""
    -   **Strategy Total Return vs. Buy & Hold:** This is the most important comparison. Did our model-driven strategy perform better than simply buying the asset at the start and holding it?
    -   **Sharpe & Calmar Ratios:** These metrics measure risk-adjusted return. A higher number is better, indicating a more efficient return for the risk taken.
    -   **Win Rate & Total Trades:** These show the consistency of the strategy. A high win rate on a good number of trades is a sign of a robust strategy.
    -   **Equity Curve & Drawdowns:** The chart visually represents the strategy's performance journey. The red shaded areas (drawdowns) show periods of loss from the portfolio's peak. A smooth, upward-sloping curve is ideal.
    """)
    
    def go_to_dashboard():
        st.session_state.page = "Strategy Dashboard"
    st.markdown("---")
    _, col2 = st.columns([4, 1]); col2.button("Go to Dashboard →", type="primary", on_click=go_to_dashboard)

def render_backtester_page():
    st.title("Strategy Dashboard"); st.sidebar.header("Strategy Configuration")
    selected_asset = st.sidebar.selectbox("1. Select Asset", list(ASSET_CONFIG.keys()), key="asset_selector")
    asset_config = ASSET_CONFIG[selected_asset]
    selected_horizon = st.sidebar.selectbox("2. Select Time Horizon", list(asset_config["horizons"].keys()), key="horizon_selector")
    selected_strategy_type = st.sidebar.radio("3. Select Strategy Type", ["Long-Only", "Long/Short"], key="type_selector")
    base_threshold = st.sidebar.slider("4. Select Base Threshold (%)", 0.05, 2.5, 0.5, 0.05, key="thresh_slider")
    
    if st.sidebar.button("Run Backtest", key="run_button"):
        with st.spinner(f"Loading model and data for {selected_asset}..."):
            model = load_model(selected_asset); predictions, target_cols = get_predictions(model, selected_asset)
        horizon_props = asset_config["horizons"][selected_horizon]
        if asset_config['timeframe_type'] == 'minute': scaled_threshold = base_threshold * (horizon_props["h"] / 1440.0)
        else: scaled_threshold = base_threshold * horizon_props["h"]
        strategy_config = {"type": selected_strategy_type, "trade_threshold": scaled_threshold, "holding_period": horizon_props["h"], "prediction_target": horizon_props["t"]}
        with st.spinner(f"Running backtest..."):
            backtest_results_tuple = run_backtest(selected_asset, predictions, target_cols, strategy_config)
        st.session_state['last_run_results'] = backtest_results_tuple
        st.session_state['last_run_params'] = {"asset": selected_asset, "horizon": selected_horizon, "strategy_type": selected_strategy_type, "scaled_threshold": scaled_threshold, "base_threshold": base_threshold}

    if 'last_run_results' in st.session_state:
        params = st.session_state['last_run_params']; results_data = st.session_state['last_run_results']
        st.header(f"Backtest Results: {params['asset']} - {params['horizon']}")
        st.markdown(f"**Strategy:** `{params['strategy_type']}` with a scaled threshold of **`{params['scaled_threshold']:.4f}%`** (base `{params['base_threshold']}%`)")
        results, equity_curve, peak, drawdown = results_data
        if results is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy Total Return", f"{results['Strategy Total Return (%)']:.2f}%"); col2.metric("Buy & Hold Return", f"{results['Buy & Hold Return (%)']:.2f}%")
            col3.metric("Annualized Sharpe Ratio", f"{results['Annualized Sharpe Ratio']:.2f}"); col4.metric("Win Rate", f"{results['Win Rate (%)']:.2f}%")
            st.subheader("Performance Metrics"); results_df = pd.DataFrame([results])
            st.dataframe(results_df.style.format("{:.2f}").set_properties(**{'text-align': 'center'}), use_container_width=True)
            st.subheader("Equity Curve & Drawdowns")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity_curve.index, equity_curve, label='Strategy Equity Curve', color='royalblue', lw=2); ax.plot(peak.index, peak, label='Running Peak', color='limegreen', ls='--', alpha=0.7); ax.fill_between(drawdown.index, equity_curve, peak, where=equity_curve<peak, color='red', alpha=0.3, label='Drawdown')
            ax.set_title(f"Equity Curve for {params['asset']} Strategy", fontsize=16); ax.set_xlabel('Trade Number'); ax.set_ylabel('Portfolio Value ($)'); ax.legend(loc='upper left'); ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("No trades were triggered for this configuration. This typically means the model's predictions did not cross the required scaled threshold. Try lowering the 'Base Threshold' slider or selecting a different time horizon.")

def render_details_page():
    st.title("Project Report: From Flawed Models to a Robust Pipeline")
    st.markdown("---"); st.header("Phase 1: The Development Journey - A Case Study on Bitcoin")
    st.markdown("""Our project began with a single goal: predict the price of Bitcoin. This initial focus served as our R&D phase, where we encountered and solved fundamental challenges that shaped our final, successful methodology.""")
    with st.expander("Attempt 1: The Data Leakage Pitfall"): st.markdown("""- **Problem:** Results were astonishingly accurate (99%+ R²), a classic sign of **data leakage** where the model "cheated". \n- **Learning:** A model must have a significant time gap between its information and its prediction.""")
    with st.expander("Attempt 2: The Persistence Trap"): st.markdown("""- **Problem:** A simple model won easily by learning the trivial "persistence" rule: `Future_Price ≈ Current_Price`. \n- **Learning:** We must predict **change** and **movement**, not absolute price levels.""")
    st.success("#### The Breakthrough: Stationarity"); st.markdown("""These failures led to the critical shift of predicting the **future percentage change**. This forced our models to learn the complex relationships between indicators that precede market movement.""")
    st.markdown("---"); st.header("Phase 2: Architecture Selection - Standard vs. Hybrid LSTM")
    st.markdown("""We tested two primary LSTM architectures on our Bitcoin data.""")
    col1, col2 = st.columns(2)
    with col1: st.subheader("Standard LSTM"); st.markdown("Given only a sequence of raw OHLCV data."); st.code("MAE for 24h: 1.70%")
    with col2: st.subheader("Hybrid LSTM"); st.markdown("Given a richer sequence of raw OHLCV data **and** technical indicators (RSI, etc.)."); st.code("MAE for 24h: 1.65%")
    st.markdown("- **Result:** The **Hybrid LSTM was demonstrably more accurate**. The engineered features provided valuable 'hints' that helped the model overcome noise.\n- **Our Champion:** We selected the **Hybrid LSTM** as our final, most powerful architecture, which is what powers this dashboard.")
    st.markdown("---"); st.header("Phase 3: Creating a Reusable, Multi-Asset Pipeline")
    st.markdown("""The final stage was to prove our methodology's robustness. The entire pipeline was successfully applied to five different assets across two different timeframes (minute and daily).""")
    st.markdown("---"); st.header("Project Assets & Notebooks")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Analysis Notebooks (.ipynb)")
        for item in [{"label": "Bitcoin (Initial LGBM)", "filename": "Bitcoin_prediction_old.ipynb"}, {"label": "Bitcoin (Final Hybrid)", "filename": "Bitcoin_pred.ipynb"}, {"label": "Ethereum (Hybrid)", "filename": "Ethereum_Pred.ipynb"}, {"label": "XPR (Hybrid)", "filename": "XPR_Pred.ipynb"}, {"label": "Solana (Hybrid)", "filename": "Solana_pred.ipynb"}, {"label": "Dogecoin (Hybrid)", "filename": "Doge_Pred.ipynb"}]:
            try:
                with open(item["filename"], "rb") as fp: st.download_button(label=f"Download: {item['label']}", data=fp, file_name=item["filename"], mime="application/x-ipynb+json", key=f"nb_{item['filename']}")
            except FileNotFoundError: st.error(f"'{item['filename']}' not found.")
    with col2:
        st.subheader("Trained Models (.pth)");
        for asset in ASSET_CONFIG:
            model_file = ASSET_CONFIG[asset]['model_path']
            try:
                with open(model_file, "rb") as fp: st.download_button(label=f"Download: {asset} Model", data=fp, file_name=model_file, mime="application/octet-stream", key=f"model_{asset}")
            except FileNotFoundError: st.error(f"'{model_file}' not found.")
    with col3:
        st.subheader("Backtest Summaries (.csv)")
        for asset_key in ["bitcoin", "ethereum", "xpr", "solana", "dogecoin"]:
            summary_file = f"{asset_key}_backtest_summary.csv"
            try:
                with open(summary_file, "rb") as fp: st.download_button(label=f"Download: {asset_key.title()} Summary", data=fp, file_name=summary_file, mime="text/csv", key=f"summary_{asset_key}")
            except FileNotFoundError: st.warning(f"'{summary_file}' not found.")

def main_app():
    st.markdown("<h1 style='text-align: center;'>Crypto Price Prediction Engine</h1>", unsafe_allow_html=True)
    load_css()
    if 'page' not in st.session_state: st.session_state.page = "Home"
    if 'current_view' not in st.session_state: st.session_state.current_view = "Home"
    
    st.sidebar.title("Navigation")
    def set_page(page_name): st.session_state.page = page_name
    st.sidebar.button("Home", on_click=set_page, args=("Home",), type="primary" if st.session_state.page == "Home" else "secondary", use_container_width=True)
    st.sidebar.button("Strategy Dashboard", on_click=set_page, args=("Strategy Dashboard",), type="primary" if st.session_state.page == "Strategy Dashboard" else "secondary", use_container_width=True)
    st.sidebar.button("Technical Details", on_click=set_page, args=("Technical Details",), type="primary" if st.session_state.page == "Technical Details" else "secondary", use_container_width=True)
    
    # --- NEW: ROBUST PAGE TRANSITION LOGIC ---
    placeholder = st.empty()
    if st.session_state.page != st.session_state.current_view:
        # If the page has changed, clear the placeholder, sleep briefly,
        # update the current view, and force a rerun.
        placeholder.empty()
        time.sleep(0.01) # A tiny delay to allow the clearing to register
        st.session_state.current_view = st.session_state.page
        st.rerun()

    with placeholder.container():
        # Render the page based on the now-stable 'current_view'
        if st.session_state.current_view == "Home":
            render_landing_page()
        elif st.session_state.current_view == "Strategy Dashboard":
            render_backtester_page()
        elif st.session_state.current_view == "Technical Details":
            render_details_page()

if __name__ == "__main__":
    main_app()