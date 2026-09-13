"""Use a portable trained model for one-step intervals and free-run centers."""
import argparse
from pathlib import Path
import numpy as np
from manfiz import MANFIZ,NARXSpec,regression_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system-folder',type=Path,default=Path('results/fresh_training/system1'))
    args=parser.parse_args(); spec=NARXSpec()
    model=MANFIZ.load(args.system_folder/'model.npz')
    with np.load(args.system_folder/'datasets.npz',allow_pickle=False) as d:
        u,y,truth=d['u_6'],d['y_6'],d['y_true_6']
    test=spec.transform(u,y)
    interval=model.predict_interval(test.X,mode='noise')
    free=spec.free_run(model,u,y[:spec.start])
    print('Measured one-step:',regression_metrics(test.y,interval.center,interval.radius))
    print('Free-run centers vs clean target:',regression_metrics(truth[spec.start:],free))


if __name__=='__main__':
    main()
