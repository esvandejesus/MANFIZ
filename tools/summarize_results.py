"""Aggregate exported metrics and render the generator-budget figure."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
from manfiz.cli import write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/fresh_training'))
    parser.add_argument('--budgets',type=Path,default=Path('results/budget_sensitivity'))
    args=parser.parse_args()
    with (args.results/'coverage_metrics.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    keys=('System','Output','Method','Amplitude','NoiseStd'); groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out=[]
    for key,parts in sorted(groups.items()):
        record=dict(zip(keys,key)); record['Replicates']=len(parts)
        for metric in ('PICP','MPIW','NMPIW','RMSE','MaxExcess','MinSlack'):
            values=[float(p[metric]) for p in parts]
            record.update({metric+'_mean':statistics.mean(values),metric+'_min':min(values),
                           metric+'_max':max(values),
                           metric+'_sd':statistics.stdev(values) if len(values)>1 else 0.})
        record['Violations_total']=sum(int(p['Violations']) for p in parts)
        out.append(record)
    write_csv(args.results/'coverage_summary.csv',out)
    if (args.budgets/'metrics.csv').exists():
        from manfiz.plotting import plot_budget_sensitivity
        with (args.budgets/'metrics.csv').open() as stream:
            plot_budget_sensitivity(list(csv.DictReader(stream)),args.budgets/'budget_sensitivity.png')
    print(f'Wrote {len(out)} scenario/channel summaries')


if __name__=='__main__':
    main()
