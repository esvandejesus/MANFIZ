"""Train one benchmark from fresh data; no pretrained artifact is read."""
import argparse
from pathlib import Path
import numpy as np
from manfiz import MANFIZ, FitConfig, PremiseConfig, NARXSpec, regression_metrics
from manfiz.benchmarks import make_training_runs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system',type=int,default=1,choices=(1,2,3))
    parser.add_argument('--max-iter',type=int,default=2000)
    parser.add_argument('--output',type=Path,default=Path('results/example_model.npz'))
    args=parser.parse_args()
    spec=NARXSpec(); runs=make_training_runs(args.system)
    train=spec.transform_runs([(r.u,r.y) for r in runs[:4]])
    validation=runs[4].regression(spec); test=runs[5].regression(spec)
    cfg=FitConfig(premise=PremiseConfig(max_iter=args.max_iter,max_evaluations=2*args.max_iter))
    model=MANFIZ(spec.input_premise_columns(3,2),config=cfg)
    model.fit(train.X,train.y,groups=train.groups,sensor_bound=np.sqrt(3)*.01)
    model.calibrate_validation(validation.X,validation.y)
    interval=model.predict_interval(test.X,mode='validation')
    print(regression_metrics(test.y,interval.center,interval.radius))
    print(model.fit_report_['premise']['message'])
    model.save(args.output)


if __name__=='__main__':
    main()
