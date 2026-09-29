import pandas as pd
from config import Config
cfg=Config();rows=[]
for k in [1,2,4,8,16,32]:
    p=cfg.eval_dir(k)/"summary.csv"
    if p.exists(): rows.append(pd.read_csv(p).iloc[0].to_dict())
if not rows: raise SystemExit("No evaluation summaries found.")
d=pd.DataFrame(rows).sort_values("n_bits")
d.to_csv(cfg.root/"capacity_curve.csv",index=False)
print(d.to_string(index=False))
print("Saved:",cfg.root/"capacity_curve.csv")
