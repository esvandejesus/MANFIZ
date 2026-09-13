"""Fit user experiments from NPZ files containing aligned arrays u and y."""
import argparse
from pathlib import Path
import numpy as np
from manfiz import MANFIZ, NARXSpec


def read_run(path):
    with np.load(path,allow_pickle=False) as data:
        return data['u'].copy(),data['y'].copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',nargs='+',type=Path,required=True,help='At least two distinct experiments')
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--sensor-bound',nargs='+',type=float,required=True,help='Deterministic amplitude, not standard deviation')
    parser.add_argument('--output',type=Path,default=Path('results/custom_model.npz'))
    args=parser.parse_args()
    runs=[read_run(p) for p in args.train]; spec=NARXSpec()
    train=spec.transform_runs(runs); validation=spec.transform(*read_run(args.validation))
    u,y=runs[0]; nu=1 if u.ndim==1 else u.shape[1]; ny=1 if y.ndim==1 else y.shape[1]
    bounds=np.asarray(args.sensor_bound)
    if len(bounds)==1:
        bounds=float(bounds[0])
    model=MANFIZ(spec.input_premise_columns(nu,ny))
    model.fit(train.X,train.y,groups=train.groups,sensor_bound=bounds)
    model.calibrate_validation(validation.X,validation.y)
    model.save(args.output)
    print(f'Saved {args.output}; use explicit clean calibration before requesting mode="noise".')


if __name__=='__main__':
    main()
