from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

try:
    from flask import Flask, jsonify, render_template_string, send_from_directory
except Exception:  # pragma: no cover - optional dependency
    Flask = None


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Experiment Runs</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
  </head>
  <body class="p-4">
    <div class="container">
      <h1>Experiment Runs</h1>
      <div id="runs"></div>
    </div>
    <script>
      async function load(){
        const r = await fetch('/api/runs');
        const data = await r.json();
        const out = document.getElementById('runs');
        if(!data.length){ out.innerHTML='<p>No runs found</p>'; return }
        let html = '<table class="table table-sm"><thead><tr><th>run_id</th><th>created_at</th><th>max_dd</th><th>sharpe</th><th>passed</th></tr></thead><tbody>';
        for(const s of data){
          html += `<tr><td><a href="/run/${s.run_id}">${s.run_id}</a></td><td>${s.created_at||''}</td><td>${s.max_drawdown||''}</td><td>${s.sharpe||''}</td><td>${s.passed||''}</td></tr>`
        }
        html += '</tbody></table>'
        out.innerHTML = html
      }
      load();
    </script>
  </body>
</html>
"""


def _collect_runs(artifacts_root: str) -> List[dict]:
    out = []
    p = Path(artifacts_root)
    if not p.exists():
        return out
    for d in sorted(p.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        sfn = d / 'summary.json'
        if not sfn.exists():
            continue
        try:
            with sfn.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            continue
        data.setdefault('run_dir', str(d))
        data.setdefault('run_id', d.name)
        out.append(data)
    return out


def create_app(artifacts_root: str = 'experiments/artifacts'):
    if Flask is None:
        raise RuntimeError('Flask not installed; install flask to run the web UI')
    app = Flask(__name__)

    @app.route('/')
    def index():
        # inject Chart.js and a small plotting helper that fetches /api/run/<id>/pnl
        script = '''
<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
<script>
    async function plot(run_id){
        const r = await fetch('/api/run/'+run_id+'/pnl');
        const data = await r.json();
        if(!data || data.length===0) return;
        const labels = data.map(d=>d[0]);
        const vals = data.map(d=>d[1]);
        const canvas = document.createElement('canvas');
        document.querySelector('.container').appendChild(canvas);
        new Chart(canvas.getContext('2d'), {type:'line', data:{labels:labels, datasets:[{label:'PnL',data:vals,fill:false,borderColor:'rgb(75,192,192)'}]}});
    }
    document.addEventListener('click', function(e){
        const a = e.target.closest('a'); if(!a) return;
        const href = a.getAttribute('href')||'';
        const parts = href.split('/');
        if(parts.length>2 && parts[1]==='run'){
            e.preventDefault();
            const runid = parts[2];
            const ex = document.querySelector('canvas'); if(ex) ex.remove();
            plot(runid);
        }
    });
</script>
'''
        return render_template_string(TEMPLATE.replace('</body>', script + '\n</body>'))

    @app.route('/api/runs')
    def api_runs():
        return jsonify(_collect_runs(artifacts_root))

    @app.route('/api/run/<run_id>/pnl')
    def api_run_pnl(run_id: str):
        p = Path(artifacts_root) / run_id
        if not p.exists():
            return jsonify({'error': 'not found'}), 404
        rfn = p / 'result.json'
        if not rfn.exists():
            return jsonify([])
        try:
            with rfn.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            return jsonify([])
        return jsonify(data.get('pnl', []))

    @app.route('/api/run/<run_id>/result')
    def api_run_result(run_id: str):
        p = Path(artifacts_root) / run_id
        if not p.exists():
            return jsonify({'error': 'not found'}), 404
        rfn = p / 'result.json'
        if not rfn.exists():
            # try to synthesize from summary if needed
            sfn = p / 'summary.json'
            if not sfn.exists():
                return jsonify({})
            try:
                with sfn.open('r', encoding='utf-8') as fh:
                    data = json.load(fh)
                return jsonify({'summary': data})
            except Exception:
                return jsonify({})
        try:
            with rfn.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            return jsonify({})
        return jsonify(data)

    @app.route('/run/<run_id>')
    def view_run(run_id: str):
        # detailed run page with charts and file list
        p = Path(artifacts_root) / run_id
        if not p.exists():
            return 'run not found', 404
        files = []
        for f in p.iterdir():
            files.append({'name': f.name, 'path': f.name})
        files_html = ''.join(["<li><a href='/run/{run_id}/file/" + f['path'] + "'>" + f['name'] + "</a></li>" for f in files])

        detail_tpl = '''
<!doctype html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>Run __RUN_ID__</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="p-4">
        <div class="container">
            <h1>Run __RUN_ID__</h1>
            <div class="mb-3">
                <button id="export_csv" class="btn btn-sm btn-outline-secondary">Export PnL CSV</button>
                <button id="export_png" class="btn btn-sm btn-outline-secondary">Download PnL PNG</button>
                <label class="ms-3">Downsample: <input id="downsample" type="number" value="0" min="0" style="width:80px"/> (0=auto)</label>
            </div>
            <div id="charts">
                <h3>PnL</h3>
                <canvas id="pnl_chart" style="max-width:100%;"></canvas>
                <h3>Drawdown</h3>
                <canvas id="dd_chart" style="max-width:100%;"></canvas>
                <h3>Executions</h3>
                <canvas id="exec_chart" style="max-width:100%;"></canvas>
            </div>
            <h3>Files</h3>
            <ul>
                __FILES_HTML__
            </ul>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            // simple downsampling: keep every step samples (n)
            function downsample(arr, target){
                if(!arr || arr.length===0) return arr;
                if(target<=0) target = 500; // default target points
                if(arr.length<=target) return arr;
                var step = Math.ceil(arr.length/target);
                var out = [];
                for(var i=0;i<arr.length;i+=step){ out.push(arr[i]); }
                // ensure last point included
                if(out[out.length-1] !== arr[arr.length-1]) out.push(arr[arr.length-1]);
                return out;
            }

            let pnlChart=null, ddChart=null, execChart=null, lastPnl=[];

            function toCSV(pnl){
                let rows = ['timestamp,value'];
                for(const r of pnl){ rows.push(`"${r[0]}",${r[1]}`); }
                return rows.join('\n');
            }

            async function loadResult(){
                const r = await fetch('/api/run/__RUN_ID__/result');
                const data = await r.json();
                const pnl = data.pnl || [];
                const execs = data.executions || [];
                lastPnl = pnl;
                function render(downN){
                    const toUse = (downN>0)? downsample(pnl, downN) : (pnl.length>1000? downsample(pnl,500): pnl);
                    const labels = toUse.map(x=>x[0]);
                    const vals = toUse.map(x=>x[1]);
                    // PnL chart
                    const ctx = document.getElementById('pnl_chart').getContext('2d');
                    if(pnlChart) pnlChart.destroy();
                    pnlChart = new Chart(ctx, {type:'line', data:{labels:labels, datasets:[{label:'PnL', data:vals, borderColor:'rgb(75,192,192)', fill:false}]}, options:{interaction:{mode:'index',intersect:false}, plugins:{tooltip:{callbacks:{label:function(ctx){return 'PnL: '+ctx.formattedValue;}}}}}});
                    // cumulative + drawdown
                    let cumvals=[]; let peak=-Infinity; let dd=[]; let s=0;
                    for(let v of vals){ s+=v; cumvals.push(s); peak = Math.max(peak,s); dd.push(peak - s); }
                    const ctx2 = document.getElementById('dd_chart').getContext('2d');
                    if(ddChart) ddChart.destroy();
                    ddChart = new Chart(ctx2, {type:'line', data:{labels:labels, datasets:[{label:'Cumulative', data:cumvals, borderColor:'green', fill:false},{label:'Drawdown', data:dd, borderColor:'red', fill:false}]}});
                    // executions
                    if(execs.length){
                        const points = execs.map(e=>({x: e.timestamp||'', y: e.price||0, size: Math.max(2, Math.log(Math.abs(e.size||1)+1)*2), info: e}));
                        const ctx3 = document.getElementById('exec_chart').getContext('2d');
                        if(execChart) execChart.destroy();
                        execChart = new Chart(ctx3, {type:'scatter', data:{datasets:[{label:'Execs', data:points, pointRadius: function(ctx){ return ctx.raw.size; }, backgroundColor:'rgba(0,123,255,0.7)'}]}, options:{plugins:{tooltip:{callbacks:{label:function(ctx){ const d=ctx.raw.info||{}; return `price:${d.price} size:${d.size} ts:${d.timestamp||''}`; }}}}, scales:{x:{type:'category'}}}});
                    }
                }

                // initial render
                render(parseInt(document.getElementById('downsample').value||0));

                // wire export buttons
                document.getElementById('export_csv').onclick = function(){ const csv=toCSV(lastPnl); const blob=new Blob([csv],{type:'text/csv'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='pnl.csv'; a.click(); };
                document.getElementById('export_png').onclick = function(){ if(!pnlChart) return; const url = pnlChart.toBase64Image(); const a=document.createElement('a'); a.href=url; a.download='pnl.png'; a.click(); };
                document.getElementById('downsample').onchange = function(){ render(parseInt(this.value||0)); };
            }
            loadResult();
        </script>
    </body>
</html>
'''
        detail_tpl = detail_tpl.replace('__RUN_ID__', run_id).replace('__FILES_HTML__', files_html)
        return detail_tpl

    @app.route('/run/<run_id>/file/<path:fname>')
    def run_file(run_id: str, fname: str):
        p = Path(artifacts_root) / run_id
        if not p.exists():
            return 'not found', 404
        return send_from_directory(str(p), fname)

    return app


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--artifacts-root', default='experiments/artifacts')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8080)
    args = p.parse_args()
    app = create_app(args.artifacts_root)
    app.run(host=args.host, port=args.port)
