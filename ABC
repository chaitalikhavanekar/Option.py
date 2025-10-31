# streamlit_app_no_matplotlib.py
"""
Illiquid Option Market Simulator (no matplotlib / no plotly).
Uses Streamlit's native plotting (st.line_chart) so it avoids matplotlib import errors.
Educational only.
"""
import streamlit as st
import numpy as np
import pandas as pd
import io
import traceback

st.set_page_config(page_title="Illiquid Option Market Simulator (No Matplotlib)", layout="wide")
st.title("Illiquid Option Market — Educational Simulator (No Matplotlib)")

st.sidebar.header("Simulation parameters")
fair_price = st.sidebar.number_input("Fair price of option", value=40.0, min_value=0.01, step=1.0)
initial_bid = st.sidebar.number_input("Initial Algo Bid", value=20.0, min_value=0.0, step=1.0)
initial_ask = st.sidebar.number_input("Initial Algo Ask", value=80.0, min_value=initial_bid+0.1, step=1.0)
time_steps = st.sidebar.slider("Time steps", min_value=10, max_value=1000, value=60, step=10)
human_order_time = st.sidebar.slider("Human arrival time (time step)", min_value=0, max_value=time_steps-1, value=2)
human_order_size = st.sidebar.number_input("Human order size (contracts)", value=1.0, min_value=0.01, step=0.1)
volatility = st.sidebar.number_input("Volatility (noise)", value=2.0, min_value=0.0, step=0.1)
mean_reversion_speed = st.sidebar.number_input("Mean reversion speed", value=0.08, min_value=0.0, max_value=1.0, step=0.01)
quote_adjust_speed = st.sidebar.number_input("Quote adjust speed", value=0.15, min_value=0.0, max_value=1.0, step=0.01)

st.sidebar.markdown("---")
st.sidebar.write("This demo is for educational purposes only. It **does not** instruct or enable market manipulation.")

run_sim = st.sidebar.button("Run simulation")

def run_simulation():
    np.random.seed(42)

    # Initialize state
    mid_price = (initial_bid + initial_ask) / 2.0
    bid = initial_bid
    ask = initial_ask

    times = []
    mid_prices = []
    bids = []
    asks = []
    events = []
    human_position = 0.0
    human_cost = 0.0

    try:
        for t in range(time_steps):
            times.append(t)
            mid_prices.append(mid_price)
            bids.append(bid)
            asks.append(ask)

            if t == human_order_time:
                # Human places a market buy -> filled at ask
                fill_price = ask
                human_position += human_order_size
                human_cost += fill_price * human_order_size
                events.append((t, f"Human market buy {human_order_size:.2f} @ {fill_price:.2f}"))

                # Algo reacts by updating its quotes toward mid_price and fair_price
                bid = bid + quote_adjust_speed * (mid_price - bid) + quote_adjust_speed * (fair_price - bid)
                ask = ask + quote_adjust_speed * (mid_price - ask) + quote_adjust_speed * (fair_price - ask)

                # Small immediate upward impact on mid_price because aggressive buy consumed ask
                mid_price = (bid + ask) / 2.0 + 0.5
            else:
                # Mean-reverting stochastic process for mid_price
                shock = np.random.normal(scale=volatility)
                mid_price += mean_reversion_speed * (fair_price - mid_price) + shock * 0.1

                # Algo gradually tightens quotes in absence of trades
                bid += quote_adjust_speed * (mid_price - bid) * 0.1
                ask += quote_adjust_speed * (mid_price - ask) * 0.1

                # Occasional external seller hits the bid (small probability)
                if np.random.rand() < 0.02:
                    event_price = bid
                    events.append((t, f"External sell hits bid @ {event_price:.2f}"))
                    mid_price = (bid + ask) / 2.0 - 0.3

            # safety bounds
            if bid < 0:
                bid = 0.01
            if ask < bid + 0.1:
                ask = bid + 0.1

        # choose a liquidation time near the end to show outcome
        liquidation_time = max(0, time_steps - 5)
        liquidation_price = bids[liquidation_time]  # human sells with market sell -> hits bid
        human_exit_pnl = human_position * liquidation_price - human_cost

        df = pd.DataFrame({
            "time": times,
            "mid_price": np.round(mid_prices, 2),
            "bid": np.round(bids, 2),
            "ask": np.round(asks, 2)
        })

        summary = {
            "Fair price": fair_price,
            "Initial bid": initial_bid,
            "Initial ask": initial_ask,
            "Human entry time": human_order_time,
            "Human entry price (avg fill)": round(human_cost / human_position, 2) if human_position else "N/A",
            "Liquidation time": liquidation_time,
            "Liquidation price (bid at liquidation)": round(liquidation_price, 2),
            "Human realized P&L": round(human_exit_pnl, 2)
        }

        return df, events, summary

    except Exception as e:
        st.error("Simulation failed. See details below.")
        st.text(traceback.format_exc())
        return None, None, None

if run_sim:
    df, events, summary = run_simulation()
    if df is None:
        st.stop()

    st.subheader("Time series — mid price and quotes (interactive via Streamlit)")
    # Use streamlit's line chart which avoids external plotting libs
    plot_df = df.set_index("time")[["mid_price", "bid", "ask"]]
    st.line_chart(plot_df)

    st.subheader("Quotes table")
    st.dataframe(df, height=300)

    st.subheader("Events")
    if events:
        for e in events:
            st.write(f"• time {e[0]}: {e[1]}")
    else:
        st.write("No trades/events recorded other than the human market buy.")

    st.subheader("Summary")
    st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]))

    # Provide CSV download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download quotes CSV", data=csv_bytes, file_name="sim_quotes.csv", mime="text/csv")

else:
    st.write("Adjust parameters on the left and click **Run simulation**.")
