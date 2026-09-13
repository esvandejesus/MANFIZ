"""Fit and calibrate a 95% quantile baseline using the new Python premises."""
import argparse
from pathlib import Path
import numpy as np
from manfiz import MANFIZ,NARXSpec
from manfiz.baselines import SharedFeatureQuantile
from manfiz.cli import metric_rows,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/fresh_training'))
    parser.add_argument('--output',type=Path,default=Path('results/quantile_comparison'))
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True); rows=[]; spec=NARXSpec()
    for folder in sorted(args.results.glob('system[123]')):
        s=int(folder.name[-1]); model=MANFIZ.load(folder/'model.npz')
        with np.load(folder/'datasets.npz',allow_pickle=False) as d:
            runs=[(d[f'u_{i}'],d[f'y_{i}']) for i in range(1,7)]
        train=spec.transform_runs(runs[:4]); val=spec.transform(*runs[4]); test=spec.transform(*runs[5])
        baseline=SharedFeatureQuantile(model).fit(train.X,train.y).calibrate(val.X,val.y)
        p=baseline.predict_interval(test.X)
        rows+=metric_rows(test.y,p.center,p.radius,System=s,Method='Shared_feature_quantile95_calibrated')
        np.savez_compressed(args.output/f'system{s}.npz',coef=baseline.coef_,calibration=baseline.calibration_,
                            calibration_rank=baseline.calibration_rank_,center=p.center,radius=p.radius)
        write_csv(args.output/'metrics.csv',rows)
        print(f'System {s}: quantile comparison complete',flush=True)


if __name__=='__main__':
    main()
