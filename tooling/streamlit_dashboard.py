"""Simple Streamlit dashboard to inspect run artifacts produced by `orchestration.paper_live`.

Usage:
  pip install streamlit
  streamlit run tooling/streamlit_dashboard.py

The app scans `experiments/artifacts` for run folders and allows selecting a run to
visualize executions, PnL, realized PnL and gate state.
"""
from __future__ import annotations

"""Enhanced Streamlit dashboard for run artifacts.

Features:
- Browse runs under `experiments/artifacts/`.
- Interactive PnL chart (line chart).
- Execution table and simple execution map (time vs price/size).
- Run log tailing (reads `run.log` if present).
- Manual refresh and simple "watch" mode (manual interval).
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import time
import altair as alt
from streamlit_autorefresh import st_autorefresh


ARTIFACTS_ROOT = Path('experiments/artifacts')


def list_runs(root: Path = ARTIFACTS_ROOT):
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def gather_recent_executions(root: Path = ARTIFACTS_ROOT, max_rows: int = 200):
    """Scan run folders for exec files and return a combined DataFrame of recent executions.

    Returns a DataFrame with a `run_id` column and normalized timestamp, side, size, price.
    """
    rows = []
    runs = list_runs(root)
    for r in runs:
        for candidate in ['execs.csv', 'executions.csv', 'exec_log.parquet', 'exec_log.csv', 'execs.parquet', 'executions.parquet']:
            p = r / candidate
            if not p.exists():
                continue
            try:
                if p.suffix == '.parquet':
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
            except Exception:
                df = None
            if df is None:
                continue

            # normalize columns
            df = df.copy()
            if 'avg_fill_price' in df.columns:
                df['price'] = df['avg_fill_price']
            elif 'exec_price' in df.columns:
                df['price'] = df['exec_price']

            if 'filled_size' in df.columns:
                df['size'] = df['filled_size']
            elif 'requested_size' in df.columns:
                df['size'] = df['requested_size']

            # timestamp normalization
            ts_col = None
            for c in ['timestamp', 'ts', 'time']:
                if c in df.columns:
                    ts_col = c
                    break
            if ts_col is not None:
                try:
                    df[ts_col] = pd.to_datetime(df[ts_col])
                    df = df.rename(columns={ts_col: 'timestamp'})
                except Exception:
                    pass

            # add run id
            df['run_id'] = r.name
            rows.append(df)
            break

    if not rows:
        return pd.DataFrame()
    try:
        combined = pd.concat(rows, ignore_index=True, sort=False)
    except Exception:
        return pd.DataFrame()

    # ensure timestamp column present and sorted
    if 'timestamp' in combined.columns:
        try:
            combined['timestamp'] = pd.to_datetime(combined['timestamp'])
            combined = combined.sort_values('timestamp', ascending=False)
        except Exception:
            pass
    else:
        # sort by index as fallback
        combined = combined.sort_index(ascending=False)

    # pick useful columns
    preferred = [c for c in ['run_id', 'pair', 'timestamp', 'side', 'size', 'price', 'order_id', 'status'] if c in combined.columns]
    if not preferred:
        return combined.head(max_rows)
    return combined[preferred].head(max_rows)


def render_run_preview(run_dir: Path, width: int = 200, height: int = 60):
    """Return an Altair chart (sparkline) for the run by reading small pnl or price series."""
    result = load_result_with_fallback(run_dir)
    pnl = result.get('pnl') or []
    if pnl:
        try:
            df = pd.DataFrame(pnl, columns=['timestamp', 'value'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            chart = alt.Chart(df).mark_line(opacity=0.8).encode(
                x='timestamp:T',
                y='value:Q'
            ).properties(width=width, height=height)
            return chart
        except Exception:
            pass

    # fallback to price_series
    ps = result.get('price_series') or []
    if ps:
        try:
            df = pd.DataFrame(ps, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            chart = alt.Chart(df).mark_line(opacity=0.8, color='#2b8cbe').encode(
                x='timestamp:T',
                y='price:Q'
            ).properties(width=width, height=height)
            return chart
        except Exception:
            pass

    # empty placeholder
    empty = pd.DataFrame({'x': [0], 'y': [0]})
    return alt.Chart(empty).mark_line().encode(x='x', y='y').properties(width=width, height=height)


def load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def load_result_with_fallback(run_dir: Path) -> dict:
    """Load result.json if present; otherwise build a minimal result from exec/pnl files."""
    res = load_json(run_dir / 'result.json') or {}
    if res:
        return res

    # Fallback: try exec_log.parquet (PaperBroker) or execs.csv/pnl.csv from run_paper_live
    exec_df = None
    for candidate in ['exec_log.parquet', 'executions.parquet', 'execs.parquet', 'execs.csv', 'executions.csv']:
        p = run_dir / candidate
        if not p.exists():
            continue
        try:
            if p.suffix == '.parquet':
                exec_df = pd.read_parquet(p)
            else:
                exec_df = pd.read_csv(p)
            break
        except Exception:
            exec_df = None

    if exec_df is not None:
        res['executions'] = exec_df.to_dict(orient='records')
        # derive price_series from executions if possible
        price_col = None
        for c in ['price', 'avg_fill_price', 'exec_price']:
            if c in exec_df.columns:
                price_col = c
                break
        if price_col is not None:
            ts_col = None
            for c in ['timestamp', 'ts', 'time']:
                if c in exec_df.columns:
                    ts_col = c
                    break
            if ts_col is not None:
                try:
                    ps_df = exec_df[[ts_col, price_col]].dropna().copy()
                    ps_df[ts_col] = pd.to_datetime(ps_df[ts_col])
                    res['price_series'] = [[t.isoformat(), float(v)] for t, v in zip(ps_df[ts_col], ps_df[price_col])]
                except Exception:
                    pass

    # Fallback: load pnl.csv
    pnl_path = run_dir / 'pnl.csv'
    if pnl_path.exists():
        try:
            pnl_df = pd.read_csv(pnl_path)
            if pnl_df.shape[1] >= 2:
                ts_col = pnl_df.columns[0]
                val_col = pnl_df.columns[1]
                pnl_df[ts_col] = pd.to_datetime(pnl_df[ts_col])
                res['pnl'] = [[t.isoformat(), float(v)] for t, v in zip(pnl_df[ts_col], pnl_df[val_col])]
            elif pnl_df.shape[1] == 1:
                val_col = pnl_df.columns[0]
                res['pnl'] = [[i, float(v)] for i, v in enumerate(pnl_df[val_col])]
        except Exception:
            pass

    # Fallback: load price_series.csv
    price_path = run_dir / 'price_series.csv'
    if price_path.exists():
        try:
            price_df = pd.read_csv(price_path)
            if price_df.shape[1] >= 2:
                ts_col = price_df.columns[0]
                val_col = price_df.columns[1]
                price_df[ts_col] = pd.to_datetime(price_df[ts_col])
                res['price_series'] = [[t.isoformat(), float(v)] for t, v in zip(price_df[ts_col], price_df[val_col])]
        except Exception:
            pass

    return res


def tail_file(p: Path, max_lines: int = 500) -> str:
    try:
        lines = p.read_text(encoding='utf-8').splitlines()
        return '\n'.join(lines[-max_lines:])
    except Exception:
        return ''


def render_executions(result: dict):
    execs = result.get('executions', []) or []
    if not execs:
        st.info('No executions recorded for this run.')
        return

    # Normalize execution fields to common names used across the repo
    df = pd.DataFrame(execs)
    # price field preference
    if 'avg_fill_price' in df.columns:
        df['price'] = df['avg_fill_price']
    elif 'exec_price' in df.columns:
        df['price'] = df['exec_price']
    # size field preference
    if 'filled_size' in df.columns:
        df['size'] = df['filled_size']
    elif 'requested_size' in df.columns:
        df['size'] = df['requested_size']

    # parse timestamps when available
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        except Exception:
            pass

    st.subheader('Executions (latest first)')
    st.dataframe(df.sort_index(ascending=False).head(200))

    # Execution map: price time series if present in result['price_series'] else small table
    price_series = result.get('price_series') or []
    if price_series:
        ps_df = pd.DataFrame(price_series, columns=['timestamp', 'price'])
        try:
            ps_df['timestamp'] = pd.to_datetime(ps_df['timestamp'])
            ps_df = ps_df.set_index('timestamp')
            st.subheader('Price Series')
            st.line_chart(ps_df['price'])
        except Exception:
            st.line_chart([p for _, p in price_series])

    # If both price_series and executions exist, plot executions as a table below the chart
    if execs and price_series:
        st.subheader('Execution Map (table)')
        # show timestamp, side, size, price (and optional order_id/pnl)
        cols = [c for c in ['timestamp', 'side', 'size', 'price', 'pnl', 'order_id'] if c in df.columns]
        st.table(df.reset_index()[cols].sort_values('timestamp', ascending=False).head(50))


def render_price_with_executions(result: dict, width: int | None = None, height: int | None = None):
    """Build an Altair layered chart: price line + execution markers.

    Expects `result` to contain `price_series` (list of [timestamp, price]) and
    `executions` (list of dicts including `timestamp`, `price`, `side`, `size`).
    """
    price_series = result.get('price_series') or []
    execs = result.get('executions', []) or []

    base_chart = None
    try:
        if price_series:
            ps_df = pd.DataFrame(price_series, columns=['timestamp', 'price'])
            ps_df['timestamp'] = pd.to_datetime(ps_df['timestamp'])
            base = alt.Chart(ps_df).mark_line(color='#1f77b4').encode(
                x=alt.X('timestamp:T', title='Time'),
                y=alt.Y('price:Q', title='Price')
            )
            base_chart = base
    except Exception:
        base_chart = None

    points = None
    try:
        if execs:
            ex_df = pd.DataFrame(execs)
            # normalize known fields used in the repo
            if 'avg_fill_price' in ex_df.columns:
                ex_df['price'] = ex_df['avg_fill_price']
            elif 'exec_price' in ex_df.columns:
                ex_df['price'] = ex_df['exec_price']
            if 'filled_size' in ex_df.columns:
                ex_df['size'] = ex_df['filled_size']
            elif 'requested_size' in ex_df.columns:
                ex_df['size'] = ex_df['requested_size']
            if 'timestamp' in ex_df.columns:
                ex_df['timestamp'] = pd.to_datetime(ex_df['timestamp'])

            if 'side' not in ex_df.columns:
                ex_df['side'] = ex_df.get('direction', 'buy')

            points = alt.Chart(ex_df).mark_point(filled=True).encode(
                x='timestamp:T',
                y='price:Q',
                color=alt.Color('side:N', scale=alt.Scale(domain=['buy','sell'], range=['#2ca02c','#d62728'])),
                size=alt.Size('size:Q', legend=None, scale=alt.Scale(range=[30, 400])),
                tooltip=['timestamp:T', 'side:N', 'size:Q', 'price:Q', 'order_id:N']
            )
    except Exception:
        points = None

    if base_chart is not None and points is not None:
        # add vertical execution rule lines and optional PnL text labels
        rule = None
        text = None
        try:
            if 'timestamp' in ex_df.columns:
                rule = alt.Chart(ex_df).mark_rule(opacity=0.6, strokeWidth=1).encode(
                    x='timestamp:T',
                    color=alt.Color('side:N', scale=alt.Scale(domain=['buy', 'sell'], range=['#2ca02c', '#d62728']))
                )
            if 'pnl' in ex_df.columns:
                # show PnL as small text above the execution marker
                text = alt.Chart(ex_df).mark_text(align='left', dx=6, dy=-12, fontSize=11).encode(
                    x='timestamp:T',
                    y='price:Q',
                    text=alt.Text('pnl:Q', format='.2f'),
                    color=alt.Color('side:N', scale=alt.Scale(domain=['buy', 'sell'], range=['#2ca02c', '#d62728']))
                )
        except Exception:
            rule = None
            text = None

        layers = [base_chart, rule, points, text]
        # filter out None
        layers = [l for l in layers if l is not None]
        layered = alt.layer(*layers).configure_axis(labelFontSize=11, titleFontSize=12)
        if width is not None:
            layered = layered.properties(width=width)
        if height is not None:
            layered = layered.properties(height=height)
        return layered
    if base_chart is not None:
        if width is not None:
            base_chart = base_chart.properties(width=width)
        if height is not None:
            base_chart = base_chart.properties(height=height)
        return base_chart
    if points is not None:
        if width is not None:
            points = points.properties(width=width)
        if height is not None:
            points = points.properties(height=height)
        return points
    return None


def main():
    st.set_page_config(page_title='Paper Live Dashboard', layout='wide')
    st.title('Paper Live — Run Artifacts Dashboard')

    # Global executions across runs
    try:
        recent_execs = gather_recent_executions(ARTIFACTS_ROOT, max_rows=300)
    except Exception:
        recent_execs = pd.DataFrame()
    with st.expander('Recent executions (all runs)', expanded=False):
        if recent_execs is None or recent_execs.empty:
            st.write('No executions found across runs.')
        else:
            # show a compact table and allow sorting/viewing
            st.dataframe(recent_execs)

    runs = list_runs()
    run_map = {r.name: r for r in runs}

    chosen = st.sidebar.selectbox('Select run', options=[''] + list(run_map.keys()))

    st.sidebar.markdown('---')
    st.sidebar.subheader('View options')
    # show recent run previews (sparklines)
    st.sidebar.markdown('**Recent runs (click Select to open)**')
    if runs:
        for r in runs[:8]:
            with st.sidebar.container():
                cols = st.columns([1, 3, 1])
                try:
                    chart = render_run_preview(r, width=160, height=50)
                except Exception:
                    chart = None
                if chart is not None:
                    cols[0].altair_chart(chart)
                cols[1].write(r.name)
                if cols[2].button('Select', key=f'select_{r.name}'):
                    st.session_state['selected_run'] = r.name

    # allow selection from session state or manual selectbox
    if 'selected_run' not in st.session_state and runs:
        # auto-select latest run for convenience
        st.session_state['selected_run'] = runs[0].name
    if 'selected_run' in st.session_state and st.session_state['selected_run'] in run_map:
        chosen = st.session_state['selected_run']

    st.sidebar.markdown('---')
    st.sidebar.subheader('Refresh / Watch')
    refresh_now = st.sidebar.button('Refresh now')
    # default to 60s auto-refresh; user can turn off or change interval
    watch = st.sidebar.checkbox('Enable watch (auto-refresh)', value=True)
    if watch:
        interval = st.sidebar.number_input('Refresh interval (seconds)', min_value=5, value=60)
        # st_autorefresh returns an incrementing int each time it fires
        _ = st_autorefresh(interval=interval * 1000, limit=None)

    if not chosen:
        st.sidebar.write('No run selected — choose a run from the list above.')
        st.sidebar.markdown('Recent runs:')
        for r in runs[:10]:
            st.sidebar.write(f"- {r.name}  (mtime: {r.stat().st_mtime})")
        return

    run_dir = run_map[chosen]
    st.sidebar.markdown(f'Run dir: `{str(run_dir)}`')

    # load artifacts (prefer full result.json, otherwise derive from exec/pnl files)
    result = load_result_with_fallback(run_dir)
    gate_state = load_json(run_dir / 'gate_state.json') or {}
    run_plan = load_json(run_dir / 'run_plan.json') or {}
    summary = load_json(run_dir / 'summary.json') or {}

    col1, col2 = st.columns([2, 1])

    with col1:
        render_executions(result)

        # Price series with executions overlay (Altair layered chart)
        layered_chart = render_price_with_executions(result, width=None, height=300)
        if layered_chart is not None:
            st.subheader('Price Series with Executions')
            st.altair_chart(layered_chart, use_container_width=True)

        # PnL chart
        pnl = result.get('pnl') or []
        if pnl:
            try:
                pnl_df = pd.DataFrame(pnl, columns=['timestamp', 'value'])
                pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
                pnl_df = pnl_df.set_index('timestamp')
                st.subheader('Marked-to-market (PnL)')
                st.line_chart(pnl_df['value'])
            except Exception:
                st.subheader('Marked-to-market (PnL)')
                st.line_chart([v for _, v in pnl])

        # Run log tailing
        st.subheader('Run log (tail)')
        log_path = run_dir / 'run.log'
        max_lines = st.number_input('Lines to show', min_value=50, max_value=5000, value=500, step=50)
        follow = st.checkbox('Follow (auto-refresh log)', value=False)
        if follow:
            # when following, refresh the page frequently so the log area updates
            _ = st_autorefresh(interval=2000, limit=None, key=f'log_follow_{run_dir.name}')

        log_text = tail_file(log_path, max_lines=max_lines)
        st.text_area('Run log (tail)', value=log_text, height=300)

        # manual refresh control for convenience
        refresh_logs = st.button('Refresh logs')
        if refresh_logs:
            # Older/newer Streamlit builds may not expose `experimental_rerun`.
            # Try to call it; if not available, update query params to force a rerun.
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    st.experimental_set_query_params(_rerun=int(time.time()))
                except Exception:
                    # Last resort: stop execution (user can manually refresh)
                    st.stop()

    with col2:
        st.subheader('Run summary')
        st.json(summary)

        st.subheader('Gate state')
        st.json(gate_state)

        st.subheader('Run plan')
        st.json(run_plan)

    # footer actions
    st.sidebar.markdown('---')
    st.sidebar.subheader('Actions')
    if refresh_now:
        st.rerun()


if __name__ == '__main__':
    main()
