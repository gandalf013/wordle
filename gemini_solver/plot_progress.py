#!/usr/bin/env python3
"""
plot_progress.py

Visualizes Wordle solver search progress over time from JSONL or terminal logs.
Generates:
1. High-resolution PNG chart (wordle_search_progress.png) if matplotlib is installed
2. Standalone interactive HTML report (wordle_search_progress.html) with zero dependencies
3. Terminal summary analytics table (throughput, slowest words, ETA)

Usage:
  python3 plot_progress.py [logfile.jsonl or logfile.txt]
  uv run --with matplotlib python3 plot_progress.py [logfile.jsonl or logfile.txt]
"""

import sys
import os
import json
import re
import math

def parse_log(filepath):
    records = []
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        return records

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSON line
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    records.append({
                        "completed": int(obj.get("completed", len(records) + 1)),
                        "total": int(obj.get("total", 14855)),
                        "word": str(obj.get("word", "")),
                        "exact_total": int(obj.get("exact_total", 0)),
                        "avg_guesses": float(obj.get("avg_guesses", 0.0)),
                        "is_exact": bool(obj.get("is_exact", False)),
                        "time_sec": float(obj.get("time_sec", 0.0)),
                        "elapsed_sec": float(obj.get("elapsed_sec", 0.0)),
                        "words_per_sec": float(obj.get("words_per_sec", 0.0)),
                        "nodes": int(obj.get("nodes", 0)),
                        "is_new_best": bool(obj.get("is_new_best", False)),
                    })
                    continue
                except Exception:
                    pass

            # Try stdout line format:
            # [ 3751/14855] Opener: meids | Status: PRUNED (>= 11412) | Time:  23.45s | Nodes: 237764
            # [    1/   10] Opener: taler | Exact Avg: 3.57837 (11483 total) | Time:   0.58s | Nodes: 5412 <-- NEW BEST
            m = re.search(r"\[\s*(\d+)/\s*(\d+)\].*?Opener:\s*([a-zA-Z]{5}).*?Time:\s*([\d\.]+)s.*?Nodes:\s*(\d+)", line)
            if m:
                idx = int(m.group(1))
                tot = int(m.group(2))
                w = m.group(3)
                t_sec = float(m.group(4))
                nodes = int(m.group(5))
                is_exact = "Exact Avg" in line
                is_new_best = "NEW BEST" in line
                exact_total = 0
                avg_guesses = 0.0
                m_exact = re.search(r"Exact Avg:\s*([\d\.]+)\s*\((\d+)\s*total\)", line)
                if m_exact:
                    avg_guesses = float(m_exact.group(1))
                    exact_total = int(m_exact.group(2))

                records.append({
                    "completed": idx,
                    "total": tot,
                    "word": w,
                    "exact_total": exact_total,
                    "avg_guesses": avg_guesses,
                    "is_exact": is_exact,
                    "time_sec": t_sec,
                    "elapsed_sec": 0.0,
                    "words_per_sec": 0.0,
                    "nodes": nodes,
                    "is_new_best": is_new_best,
                })

    # Compute cumulative elapsed times if not present
    cum_time = 0.0
    for r in records:
        if r["elapsed_sec"] <= 0.0:
            cum_time += r["time_sec"]
            r["elapsed_sec"] = cum_time
        if r["words_per_sec"] <= 0.0 and r["elapsed_sec"] > 0.001:
            r["words_per_sec"] = r["completed"] / r["elapsed_sec"]

    return records

def print_summary(records):
    if not records:
        print("No valid search progress records found.")
        return

    n = len(records)
    tot = records[-1]["total"]
    total_elapsed = records[-1]["elapsed_sec"]
    total_nodes = sum(r["nodes"] for r in records)
    total_exact = sum(1 for r in records if r["is_exact"])
    total_pruned = n - total_exact
    avg_time_per_word = (total_elapsed / n) if n > 0 else 0.0
    avg_nodes_per_word = (total_nodes / n) if n > 0 else 0

    print("\n" + "=" * 65)
    print("           WORDLE SOLVER PROGRESS SUMMARY")
    print("=" * 65)
    print(f" Words Processed:       {n:,} / {tot:,} ({n*100.0/tot:.1f}%)")
    print(f" Elapsed Wall-Clock:    {total_elapsed:.1f} s ({total_elapsed/60.0:.2f} min / {total_elapsed/3600.0:.2f} hrs)")
    print(f" Total Nodes Visited:   {total_nodes:,}")
    print(f" Exact Solved Words:    {total_exact:,} ({total_exact*100.0/n:.1f}%)")
    wps_str = f"{n / total_elapsed:.2f} words/sec" if total_elapsed > 0 else "N/A"
    nps_str = f"{total_nodes / total_elapsed:.0f} nodes/sec" if total_elapsed > 0 else "N/A"
    print(f" Overall Throughput:    {wps_str} ({nps_str})")
    print(f" Average Time / Word:   {avg_time_per_word:.4f} s")
    print("-" * 65)

    # Slowest words (Bottlenecks)
    sorted_by_time = sorted(records, key=lambda x: x["time_sec"], reverse=True)
    print(" Top 10 Slowest Words (Near-Miss Bottlenecks):")
    print("  Rank | Word  | Time (s) | Nodes Visited | Status")
    print(" ------+-------+----------+---------------+------------------")
    for i, r in enumerate(sorted_by_time[:10]):
        status = f"Exact ({r['exact_total']})" if r["is_exact"] else "PRUNED"
        print(f"  {i+1:4d} | {r['word']:<5s} | {r['time_sec']:8.2f} | {r['nodes']:13,d} | {status}")
    print("=" * 65 + "\n")

def generate_html_report(records, output_path="wordle_search_progress.html"):
    if not records:
        return

    # Downsample points for smooth web rendering if > 2000 records
    sample_step = max(1, len(records) // 1500)
    sampled = records[::sample_step]
    if sampled[-1] != records[-1]:
        sampled.append(records[-1])

    indices = [r["completed"] for r in sampled]
    times = [round(r["time_sec"], 4) for r in sampled]
    nodes = [r["nodes"] for r in sampled]
    elapsed_min = [round(r["elapsed_sec"] / 60.0, 2) for r in sampled]
    wps = [round(r["words_per_sec"], 2) for r in sampled]
    words = [r["word"] for r in sampled]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wordle Solver Search Progress</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 24px;
      font-weight: 700;
      margin-bottom: 4px;
      color: #38bdf8;
    }}
    .subtitle {{
      color: #94a3b8;
      font-size: 14px;
      margin-bottom: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 18px;
    }}
    .card h2 {{
      font-size: 16px;
      color: #e2e8f0;
      margin-top: 0;
      margin-bottom: 12px;
    }}
    .chart-container {{
      position: relative;
      height: 320px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Wordle Minimax Solver &mdash; Search Progress Analysis</h1>
    <div class="subtitle">Completed {len(records):,} / {records[-1]['total']:,} words in {records[-1]['elapsed_sec']/60.0:.1f} minutes &bull; Shared Transposition Table Scaling</div>

    <div class="grid">
      <div class="card">
        <h2>Time per Word (Seconds) vs. Word Index</h2>
        <div class="chart-container">
          <canvas id="timeChart"></canvas>
        </div>
      </div>
      <div class="card">
        <h2>Nodes Explored per Word vs. Word Index</h2>
        <div class="chart-container">
          <canvas id="nodesChart"></canvas>
        </div>
      </div>
      <div class="card">
        <h2>Cumulative Elapsed Time (Minutes)</h2>
        <div class="chart-container">
          <canvas id="elapsedChart"></canvas>
        </div>
      </div>
      <div class="card">
        <h2>Solver Throughput (Words / Second)</h2>
        <div class="chart-container">
          <canvas id="throughputChart"></canvas>
        </div>
      </div>
    </div>
  </div>

  <script>
    const indices = {json.dumps(indices)};
    const times = {json.dumps(times)};
    const nodes = {json.dumps(nodes)};
    const elapsed = {json.dumps(elapsed_min)};
    const wps = {json.dumps(wps)};
    const words = {json.dumps(words)};

    const commonOptions = {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            title: (items) => `Word #${{indices[items[0].dataIndex]}}: ${{words[items[0].dataIndex]}}`
          }}
        }}
      }},
      scales: {{
        x: {{
          grid: {{ color: '#334155' }},
          ticks: {{ color: '#94a3b8' }},
          title: {{ display: true, text: 'Word Pre-Rank Index', color: '#94a3b8' }}
        }},
        y: {{
          grid: {{ color: '#334155' }},
          ticks: {{ color: '#94a3b8' }}
        }}
      }}
    }};

    new Chart(document.getElementById('timeChart'), {{
      type: 'line',
      data: {{
        labels: indices,
        datasets: [{{
          label: 'Time (s)',
          data: times,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true
        }}]
      }},
      options: commonOptions
    }});

    new Chart(document.getElementById('nodesChart'), {{
      type: 'line',
      data: {{
        labels: indices,
        datasets: [{{
          label: 'Nodes',
          data: nodes,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true
        }}]
      }},
      options: commonOptions
    }});

    new Chart(document.getElementById('elapsedChart'), {{
      type: 'line',
      data: {{
        labels: indices,
        datasets: [{{
          label: 'Elapsed (min)',
          data: elapsed,
          borderColor: '#10b981',
          borderWidth: 2,
          pointRadius: 0
        }}]
      }},
      options: commonOptions
    }});

    new Chart(document.getElementById('throughputChart'), {{
      type: 'line',
      data: {{
        labels: indices,
        datasets: [{{
          label: 'Words / Sec',
          data: wps,
          borderColor: '#a855f7',
          backgroundColor: 'rgba(168, 85, 247, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true
        }}]
      }},
      options: commonOptions
    }});
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated interactive HTML report: {output_path}")

def generate_png_plot(records, output_path="wordle_search_progress.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    indices = [r["completed"] for r in records]
    times = [r["time_sec"] for r in records]
    nodes = [r["nodes"] for r in records]
    elapsed = [r["elapsed_sec"] / 60.0 for r in records]
    wps = [r["words_per_sec"] for r in records]

    # Rolling average for time
    window = max(5, len(records) // 100)
    rolling_time = []
    for i in range(len(times)):
        start = max(0, i - window + 1)
        rolling_time.append(sum(times[start:i+1]) / (i - start + 1))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0f172a')

    for ax in axs.flat:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f8fafc')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Panel 1: Time per Word
    axs[0, 0].plot(indices, times, color='#38bdf8', alpha=0.3, label='Raw Time')
    axs[0, 0].plot(indices, rolling_time, color='#0284c7', linewidth=2, label=f'Moving Avg ({window})')
    axs[0, 0].set_title('Computation Time per Word (Seconds)')
    axs[0, 0].set_xlabel('Word Pre-Rank Index')
    axs[0, 0].set_ylabel('Seconds')
    axs[0, 0].legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')

    # Panel 2: Nodes Visited
    axs[0, 1].plot(indices, nodes, color='#f59e0b', linewidth=1)
    axs[0, 1].set_title('Nodes Explored per Word')
    axs[0, 1].set_xlabel('Word Pre-Rank Index')
    axs[0, 1].set_ylabel('Nodes')

    # Panel 3: Cumulative Time
    axs[1, 0].plot(indices, elapsed, color='#10b981', linewidth=2)
    axs[1, 0].set_title('Cumulative Elapsed Time (Minutes)')
    axs[1, 0].set_xlabel('Word Pre-Rank Index')
    axs[1, 0].set_ylabel('Minutes')

    # Panel 4: Word Throughput
    axs[1, 1].plot(indices, wps, color='#a855f7', linewidth=2)
    axs[1, 1].set_title('Overall Throughput (Words / Second)')
    axs[1, 1].set_xlabel('Word Pre-Rank Index')
    axs[1, 1].set_ylabel('Words / Sec')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated high-resolution plot: {output_path}")

def main():
    default_files = ["results.jsonl", "log.txt", "wordle.log", "output.txt"]
    filepath = None
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        for f in default_files:
            if os.path.exists(f):
                filepath = f
                break

    if not filepath:
        print("Usage: python3 plot_progress.py <results.jsonl or log.txt>")
        return

    records = parse_log(filepath)
    if not records:
        print(f"No records parsed from '{filepath}'.")
        return

    print_summary(records)
    generate_html_report(records, "wordle_search_progress.html")
    generate_png_plot(records, "wordle_search_progress.png")

if __name__ == "__main__":
    main()
