from __future__ import annotations

import json
import math
import struct
from datetime import date, timedelta
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR.parent / ".qlib" / "qlib_data" / "cn_data"
OUTPUT_PATH = EXAMPLES_DIR / "output" / "a_share_data_quick_test.html"
SYMBOL = "SH600519"
SYMBOL_NAME = "贵州茅台"


def read_feature(symbol: str, field: str) -> tuple[int, list[float]]:
    path = DATA_DIR / "features" / symbol.lower() / f"{field}.day.bin"
    payload = path.read_bytes()
    values = struct.unpack(f"<{len(payload) // 4}f", payload)
    return int(values[0]), list(values[1:])


def load_market_data(days: int = 365) -> list[dict[str, float | str]]:
    calendar = (DATA_DIR / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
    cutoff = date.fromisoformat(calendar[-1]) - timedelta(days=days)
    fields = {}
    start_indices = set()
    for field in ("open", "high", "low", "close", "volume"):
        start_index, values = read_feature(SYMBOL, field)
        start_indices.add(start_index)
        fields[field] = values
    if len(start_indices) != 1:
        raise ValueError(f"Feature start indices do not match: {sorted(start_indices)}")

    start_index = start_indices.pop()
    records = []
    for offset in range(min(map(len, fields.values()))):
        trading_date = date.fromisoformat(calendar[start_index + offset])
        values = {field: fields[field][offset] for field in fields}
        if trading_date >= cutoff and all(math.isfinite(value) for value in values.values()):
            records.append({"date": trading_date.isoformat(), **values})
    if not records:
        raise ValueError(f"No finite records found for {SYMBOL}")
    if any(record["low"] > min(record["open"], record["close"]) for record in records):
        raise ValueError("Invalid OHLC data: low exceeds open or close")
    if any(record["high"] < max(record["open"], record["close"]) for record in records):
        raise ValueError("Invalid OHLC data: high is below open or close")
    return records


def build_html(records: list[dict[str, float | str]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    latest = records[-1]
    first_close = float(records[0]["close"])
    last_close = float(latest["close"])
    change = last_close / first_close - 1
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>A股数据快速测试</title>
  <style>
    :root {{ color-scheme: light; --ink:#18221d; --muted:#68746d; --paper:#f4f1e8; --panel:#fffdf7; --grid:#d9d6cb; --up:#b42318; --down:#087f5b; --accent:#b7791f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:linear-gradient(135deg,#ece8dc 0%,#f8f6ef 48%,#e5ebe5 100%); color:var(--ink); font-family:"Microsoft YaHei UI","Segoe UI",sans-serif; }}
    main {{ width:min(1180px,calc(100% - 28px)); margin:24px auto; }}
    header {{ display:flex; align-items:end; justify-content:space-between; gap:20px; margin-bottom:16px; }}
    h1 {{ margin:0; font-family:Georgia,"Microsoft YaHei UI",serif; font-size:clamp(25px,4vw,44px); letter-spacing:0; }}
    .subtitle,.source {{ color:var(--muted); font-size:13px; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); border-block:1px solid #c8c5bb; margin-bottom:18px; }}
    .stat {{ padding:13px 16px; border-right:1px solid #c8c5bb; }} .stat:last-child {{ border:0; }}
    .label {{ color:var(--muted); font-size:12px; }} .value {{ margin-top:5px; font-family:Georgia,serif; font-size:21px; font-variant-numeric:tabular-nums; }}
    .chart {{ background:rgba(255,253,247,.9); border:1px solid #c8c5bb; box-shadow:0 12px 32px rgba(40,48,42,.09); padding:12px; }}
    svg {{ display:block; width:100%; height:auto; min-height:460px; }}
    .axis {{ fill:var(--muted); font-size:11px; }} .grid {{ stroke:var(--grid); stroke-width:1; }}
    .wick {{ stroke-width:1; }} .candle {{ stroke-width:1; }} .volume {{ opacity:.32; }}
    .crosshair {{ stroke:#6b705c; stroke-width:1; stroke-dasharray:4 4; pointer-events:none; }}
    .tooltip {{ position:fixed; display:none; pointer-events:none; background:#18221d; color:#fffdf7; padding:9px 11px; font-size:12px; line-height:1.65; border-radius:4px; box-shadow:0 8px 22px rgba(0,0,0,.2); font-variant-numeric:tabular-nums; }}
    footer {{ display:flex; justify-content:space-between; gap:16px; margin-top:10px; }}
    @media (max-width:700px) {{ header,footer {{ align-items:start; flex-direction:column; }} .stats {{ grid-template-columns:1fr 1fr; }} .stat:nth-child(2) {{ border-right:0; }} .stat:nth-child(-n+2) {{ border-bottom:1px solid #c8c5bb; }} svg {{ min-height:400px; }} }}
  </style>
</head>
<body>
<main>
  <header><div><div class="subtitle">QLIB MARKET DATA / QUICK CHECK</div><h1>{SYMBOL_NAME} · {SYMBOL}</h1></div><div class="source">近一年 · 前复权行情</div></header>
  <section class="stats">
    <div class="stat"><div class="label">最新日期</div><div class="value">{latest['date']}</div></div>
    <div class="stat"><div class="label">最新收盘</div><div class="value">{last_close:.2f}</div></div>
    <div class="stat"><div class="label">区间涨跌</div><div class="value" style="color:var({'--up' if change >= 0 else '--down'})">{change:+.2%}</div></div>
    <div class="stat"><div class="label">有效交易日</div><div class="value">{len(records)}</div></div>
  </section>
  <section class="chart"><svg id="chart" viewBox="0 0 1120 620" role="img" aria-label="贵州茅台日K线与成交量图"></svg></section>
  <footer><span class="source">红涨绿跌 · 移动指针查看 OHLCV</span><span class="source">数据目录：.qlib/qlib_data/cn_data</span></footer>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const data={payload};
const svg=document.getElementById('chart'),tip=document.getElementById('tooltip');
const NS='http://www.w3.org/2000/svg',W=1120,H=620,M={{l:66,r:24,t:22,b:36}},priceH=430,gap=24,volumeTop=476,volumeH=95;
const prices=data.flatMap(d=>[d.low,d.high]),pMin=Math.min(...prices),pMax=Math.max(...prices),pad=(pMax-pMin)*.06;
const lo=pMin-pad,hi=pMax+pad,maxVol=Math.max(...data.map(d=>d.volume));
const x=i=>M.l+(i+.5)*(W-M.l-M.r)/data.length, y=v=>M.t+(hi-v)/(hi-lo)*priceH;
const node=(tag,attrs={{}})=>{{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));svg.appendChild(n);return n;}};
for(let i=0;i<=5;i++){{const py=M.t+i*priceH/5,val=hi-i*(hi-lo)/5;node('line',{{x1:M.l,y1:py,x2:W-M.r,y2:py,class:'grid'}});const t=node('text',{{x:8,y:py+4,class:'axis'}});t.textContent=val.toFixed(0);}}
for(let i=0;i<=4;i++){{const idx=Math.min(data.length-1,Math.round(i*(data.length-1)/4)),px=x(idx);node('line',{{x1:px,y1:M.t,x2:px,y2:volumeTop+volumeH,class:'grid'}});const t=node('text',{{x:px-28,y:H-10,class:'axis'}});t.textContent=data[idx].date.slice(5);}}
const width=Math.max(1.5,(W-M.l-M.r)/data.length*.62);
data.forEach((d,i)=>{{const px=x(i),up=d.close>=d.open,color=up?'var(--up)':'var(--down)';node('line',{{x1:px,y1:y(d.high),x2:px,y2:y(d.low),stroke:color,class:'wick'}});node('rect',{{x:px-width/2,y:Math.min(y(d.open),y(d.close)),width,height:Math.max(1,Math.abs(y(d.open)-y(d.close))),fill:up?color:'var(--panel)',stroke:color,class:'candle'}});const vh=d.volume/maxVol*volumeH;node('rect',{{x:px-width/2,y:volumeTop+volumeH-vh,width,height:vh,fill:color,class:'volume'}});}});
const cross=node('line',{{y1:M.t,y2:volumeTop+volumeH,class:'crosshair',visibility:'hidden'}});
const hit=node('rect',{{x:M.l,y:M.t,width:W-M.l-M.r,height:volumeTop+volumeH-M.t,fill:'transparent'}});
hit.addEventListener('mousemove',e=>{{const r=svg.getBoundingClientRect(),sx=(e.clientX-r.left)/r.width*W,idx=Math.max(0,Math.min(data.length-1,Math.floor((sx-M.l)/(W-M.l-M.r)*data.length))),d=data[idx];cross.setAttribute('x1',x(idx));cross.setAttribute('x2',x(idx));cross.setAttribute('visibility','visible');tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-185)+'px';tip.style.top=Math.max(8,e.clientY-95)+'px';tip.innerHTML=`<b>${{d.date}}</b><br>开 ${{d.open.toFixed(2)}}　高 ${{d.high.toFixed(2)}}<br>低 ${{d.low.toFixed(2)}}　收 ${{d.close.toFixed(2)}}<br>量 ${{(d.volume/1e6).toFixed(2)}} 百万`;}});
hit.addEventListener('mouseleave',()=>{{cross.setAttribute('visibility','hidden');tip.style.display='none';}});
</script>
</body>
</html>"""


def main() -> None:
    records = load_market_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_html(records), encoding="utf-8")
    print(f"Validated {len(records)} daily records for {SYMBOL}")
    print(f"Date range: {records[0]['date']} -> {records[-1]['date']}")
    print(f"Chart written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()