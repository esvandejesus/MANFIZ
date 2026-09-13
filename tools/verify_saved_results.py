"""Independent checks on saved fresh-training models, logs and summary tables."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from manfiz import MANFIZ,NARXSpec,regression_metrics
from manfiz.premises import design_matrix,evaluate_center
from manfiz.io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/fresh_training'))
    args=parser.parse_args(); root=args.results; records=[]; spec=NARXSpec()
    for folder in sorted(root.glob('system[123]')):
        model=MANFIZ.load(folder/'model.npz')
        with np.load(folder/'datasets.npz',allow_pickle=False) as data:
            runs=[(data[f'u_{i}'],data[f'y_{i}']) for i in range(1,7)]
        train=spec.transform_runs(runs[:4]); test=spec.transform(*runs[5])
        phi,w=model.features(train.X); h=design_matrix(phi,w)
        residual=model.y_scaler_.transform(train.y)-evaluate_center(phi,w,model.beta_)
        witness_errors=[]
        for o,lp in enumerate(model.feasibility_):
            delta=np.asarray(lp['witness'])
            error=float(abs(residual[:,o]-h@delta).max()*model.y_scaler_.scale[o])
            assert error <= model.regression_bound_[o]+1e-9
            witness_errors.append(error)
        p=model.predict_interval(train.X,mode='parameters')
        met=regression_metrics(train.y,p.center,p.radius+model.regression_bound_)
        assert np.all(met['Violations']==0)
        with np.load(folder/'heldout_predictions.npz',allow_pickle=False) as saved:
            current=model.predict_interval(test.X,mode='noise')
            assert np.array_equal(current.center,saved['center'])
            assert np.array_equal(current.radius,saved['radius'])
        posterior_cap=model.config.reduction.q_max+model.config.batch_size
        logs=[log for o in model.outputs_ for log in o['history']]
        assert all(log['posterior_columns'] <= posterior_cap for log in logs)
        feasible=[log for log in logs if log['triggered'] and not log['fallback']]
        assert all(log['cycle_ratio'] <= model.config.reduction.max_cycle_ratio*(1+model.config.reduction.cycle_tolerance)+1e-12 for log in feasible)
        records.append(dict(System=int(folder.name[-1]),TrainingChannelValues=train.y.size,
                            TrainingViolations=int(met['Violations'].sum()),
                            TrainingMinSlack=float(met['MinSlack'].min()),
                            CommonWitnessMaxPhysicalResidual=witness_errors,
                            AcceptedUpdates=len(logs),PosteriorCap=posterior_cap,
                            CapViolations=0,FeasibleReductionEvents=len(feasible),
                            FeasibleCycleViolations=0,StoredPredictionsExact=True))
    summary=json.loads((root/'summary.json').read_text())
    with (root/'coverage_metrics.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['Method']=='MANFIZ_noise']
    assert sum(int(r['Violations']) for r in rows)==summary['noisy_violations']
    assert len(rows)*(json.loads((root/'protocol.json').read_text())['n_samples']-spec.start)==summary['measured_channel_values']
    write_json(root/'independent_verification.json',dict(systems=records,
                confirmation_count_and_violations_consistent=True,
                original_experiment_summary=summary))
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
