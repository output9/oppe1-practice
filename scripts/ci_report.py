import json
from pathlib import Path
import pandas as pd

def main():
    lines = ["# CI Report\n"]
    s = Path("data/processed/v0/summary.json")
    if s.exists():
        j = json.loads(s.read_text())
        lines += [
            "## Preprocessing Summary",
            f"- Versions: `{'+'.join(j['versions'])}`",
            f"- Rows: total={j['rows_total']} train={j['rows_train']} test={j['rows_test']}",
            f"- Stocks: {', '.join(j['stocks'])}",
            "",
        ]
    p = Path("outputs/preds_v0.csv")
    if p.exists():
        df = pd.read_csv(p)
        lines += ["## Predictions preview", df.head(10).to_markdown(index=False)]
    Path("ci_report.md").write_text("\n".join(lines))

if __name__ == "__main__":
    main()
