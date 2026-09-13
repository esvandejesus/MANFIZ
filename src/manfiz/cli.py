"""Reproduce the native Python training and independent noise-coverage study."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from . import MANFIZ, FitConfig, PremiseConfig, ReductionConfig, NARXSpec, __version__
from .benchmarks import make_run, make_training_runs
from .metrics import regression_metrics
from .io import write_json, fingerprint, jsonable


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader(); writer.writerows([jsonable(r) for r in rows])


def metric_rows(y, center, radius=None, **labels):
    met = regression_metrics(y,center,radius)
    return [dict(**labels,Output=o+1,**{k:v[o] for k,v in met.items()}) for o in range(y.shape[1])]


def source_hashes():
    root = Path(__file__).parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def run_benchmarks(output, *, systems=(1,2,3), config=None, n_samples=1800,
                   calibration_replicates=20, confirmation_replicates=20, seed=20260913,
                   calibration_seed=326091300, confirmation_seed=426091300, plots=True):
    """Train new models; freeze all bounds before evaluating confirmation data."""
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('Choose a new output directory; existing experiments are not overwritten')
    if (not systems or len(set(systems)) != len(systems) or any(s not in (1,2,3) for s in systems)
        or calibration_replicates < 1 or confirmation_replicates < 1 or n_samples <= 130):
        raise ValueError('Invalid system selection, trajectory length or replicate count')
    if calibration_seed == confirmation_seed:
        raise ValueError('Calibration and confirmation seed bases must differ')
    config = FitConfig() if config is None else config
    spec = NARXSpec(); config.validate(11)
    protocol = dict(version=__version__,systems=list(systems),config=asdict(config),n_samples=n_samples,
                    discard=spec.discard,training_runs=4,validation_runs=1,heldout_runs=1,
                    seed=seed,calibration_seed=calibration_seed,confirmation_seed=confirmation_seed,
                    calibration_replicates=calibration_replicates,
                    confirmation_replicates=confirmation_replicates,
                    amplitudes=[1.,1.15,1.30],noise_standard_deviations=[.005,.01,.02],
                    training_sensor_bound=float(np.sqrt(3)*.01),
                    frozen_evaluation_sensor_bound=float(np.sqrt(3)*.02),
                    rng='NumPy default_rng/PCG64', initialization='Fresh premise grid; no pretrained files read',
                    calibration_safety=1.10,validation_safety=1.05,
                    pairing='Phase/noise seeds paired across amplitude/noise conditions; calibration and confirmation use separate bases',
                    coverage_tolerance=1e-10,created_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
                    blas_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
                    source_sha256=source_hashes())
    write_json(output/'protocol.json',protocol)  # committed locally before looking at test outcomes
    heldout_rows=[]; coverage_rows=[]; clean_rows=[]; free_rows=[]; hashes=[]; training=[]
    started=time.perf_counter()
    for s in systems:
        folder=output/f'system{s}'; folder.mkdir(exist_ok=True)
        runs=make_training_runs(s,n_samples=n_samples,seed=seed)
        train=spec.transform_runs([(r.u,r.y) for r in runs[:4]])
        val=runs[4].regression(spec); test=runs[5].regression(spec)
        arrays={f'{key}_{r}':getattr(run,key) for r,run in enumerate(runs,1) for key in ('u','y','y_true','phase')}
        arrays['noise_seeds']=np.array([r.noise_seed for r in runs])
        np.savez_compressed(folder/'datasets.npz',**arrays)
        print(f'System {s}: training shared premises from scratch on {len(train.X)} rows',flush=True)
        model=MANFIZ(spec.input_premise_columns(3,2),config=config)
        model.fit_nominal(train.X,train.y)
        pr=model.fit_report_['premise']
        print(f"System {s}: objective {pr['initial_objective']:.8g} -> {pr['final_objective']:.8g}; {pr['message']}",flush=True)
        model.fit_zonotopes(train.X,train.y,groups=train.groups,sensor_bound=np.sqrt(3)*.01)
        model.calibrate_validation(val.X,val.y)
        write_csv(folder/'optimizer_history.csv',
                  [dict(Evaluation=i,Objective=r[0],DataMSE=r[1],Penalty=r[2])
                   for i,r in enumerate(pr['objective_history'])])
        for o,out in enumerate(model.outputs_,1):
            write_json(folder/f'updates_y{o}.json',out['history'])
        print(f"System {s}: accepted updates {model.fit_report_['accepted_updates']}; calibrating clean envelope",flush=True)
        clean_parts=[]
        for amp in protocol['amplitudes']:
            for rep in range(1,calibration_replicates+1):
                ps=calibration_seed+1000*s+rep
                run=make_run(s,n_samples=n_samples,amplitude=amp,noise_std=0,
                             phase_seed=ps,noise_seed=ps+100000)
                part=run.regression(spec,clean=True); clean_parts.append(part)
                hashes.append(dict(System=s,Stage='clean_calibration',Amplitude=amp,NoiseStd=0,
                                   Replicate=rep,PhaseSeed=ps,NoiseSeed=ps+100000,
                                   SHA256=fingerprint(run.u,run.y)))
        bx=spec.regressor_noise_bounds(3,np.full(2,np.sqrt(3)*.02))
        model.calibrate_noise(np.concatenate([d.X for d in clean_parts]),
                              np.concatenate([d.y for d in clean_parts]),
                              sensor_bound=np.sqrt(3)*.02,regressor_bound=bx)
        model.save(folder/'model.npz')
        reloaded=MANFIZ.load(folder/'model.npz')
        original=model.predict_interval(test.X,mode='noise'); restored=reloaded.predict_interval(test.X,mode='noise')
        if not (np.array_equal(original.center,restored.center) and np.array_equal(original.radius,restored.radius)):
            raise AssertionError('Save/load changed predictions')
        np.savez_compressed(folder/'heldout_predictions.npz',y=test.y,y_true=runs[5].y_true[spec.start:],
                            sample_indices=test.sample_indices,center=original.center,radius=original.radius,
                            **{f'radius_{k}':v for k,v in original.components.items()})
        for mode in ('parameters','validation','noise'):
            p=model.predict_interval(test.X,mode=mode)
            heldout_rows+=metric_rows(test.y,p.center,p.radius,System=s,Method=f'MANFIZ_{mode}')
        nom=model.predict(test.X,nominal=True)
        nominal_box=1.05*abs(val.y-model.predict(val.X,nominal=True)).max(0)
        heldout_rows+=metric_rows(test.y,nom,nominal_box,System=s,Method='MANFIS_validation_box')
        for nominal in (True,False):
            free=spec.free_run(model,runs[5].u,runs[5].y[:spec.start],nominal=nominal)
            one=model.predict(test.X,nominal=nominal)
            for mode,pred in [('free_run',free),('measured_lag_one_step',one)]:
                free_rows+=metric_rows(runs[5].y_true[spec.start:],pred,System=s,
                                       Method='MANFIS' if nominal else 'MANFIZ',Mode=mode)
        for amp in protocol['amplitudes']:
            for rep in range(1,confirmation_replicates+1):
                ps=confirmation_seed+1000*s+rep
                # Clean validity is checked independently, without changing the frozen model.
                base=make_run(s,n_samples=n_samples,amplitude=amp,noise_std=0,
                              phase_seed=ps,noise_seed=ps+100000)
                dc=base.regression(spec,clean=True)
                pc=model.predict_interval(dc.X,mode='parameters')
                clean_rows+=metric_rows(dc.y,pc.center,pc.radius+model.noise_calibration_['clean_defect'],
                                        System=s,Amplitude=amp,Replicate=rep)
                for std in protocol['noise_standard_deviations']:
                    # Pair the same bounded noise pattern across standard deviations.
                    noise=np.sqrt(3)*std*np.random.default_rng(ps+100000).uniform(-1,1,base.y.shape)
                    ym=base.y_true+noise; data=spec.transform(base.u,ym)
                    p=model.predict_interval(data.X,mode='noise')
                    labels=dict(System=s,Amplitude=amp,NoiseStd=std,Replicate=rep)
                    coverage_rows+=metric_rows(data.y,p.center,p.radius,Method='MANFIZ_noise',**labels)
                    coverage_rows+=metric_rows(data.y,p.center,p.components['parameters']+model.validation_bound_,
                                               Method='MANFIZ_validation',**labels)
                    hashes.append(dict(System=s,Stage='confirmation',Amplitude=amp,NoiseStd=std,
                                       Replicate=rep,PhaseSeed=ps,NoiseSeed=ps+100000,
                                       SHA256=fingerprint(base.u,ym)))
            print(f'System {s}: independent confirmation amplitude {amp} finished',flush=True)
        record=dict(System=s,**{k:v for k,v in pr.items() if k not in ('objective_history',)},
                    regression_bound=model.regression_bound_,
                    minimum_joint_error=[a['minimum_error_physical'] for a in model.feasibility_],
                    sensor_only_feasible=[a['sensor_only_training_feasible'] for a in model.feasibility_],
                    lp_dual_gap=[a['dual_gap'] for a in model.feasibility_],
                    accepted_updates=model.fit_report_['accepted_updates'],
                    witness_max_error=[max(o.get('witness_membership_errors',[np.nan])) for o in model.outputs_],
                    clean_defect=model.noise_calibration_['clean_defect'],
                    posterior_max_columns=max(r.shape[1] for o in model.outputs_ for r in o['generators']),
                    reduction_events=sum(log['triggered'] for o in model.outputs_ for log in o['history']),
                    reduction_fallbacks=sum(log['fallback'] for o in model.outputs_ for log in o['history']),
                    save_load_bitwise_equal=True)
        training.append(record); write_json(folder/'training_report.json',record)
        for name,rows in [('heldout_metrics',heldout_rows),('coverage_metrics',coverage_rows),
                          ('clean_confirmation_metrics',clean_rows),('free_run_metrics',free_rows),
                          ('trajectory_hashes',hashes)]:
            write_csv(output/f'{name}.csv',rows)
        write_json(output/'training_report.json',training)
        if plots:
            from .plotting import plot_training, plot_intervals
            plot_training(pr,folder/'training.png')
            plot_intervals(test.sample_indices,test.y,original,folder/'intervals.png',system=s)
    rows=[r for r in coverage_rows if r['Method']=='MANFIZ_noise']
    summary=dict(systems=list(systems),confirmation_trajectories=len(rows)//2,
                 measured_channel_values=len(rows)*(n_samples-spec.start),
                 noisy_violations=int(sum(r['Violations'] for r in rows)),
                 minimum_noisy_slack=float(min(r['MinSlack'] for r in rows)),
                 clean_violations=int(sum(r['Violations'] for r in clean_rows)),
                 minimum_clean_slack=float(min(r['MinSlack'] for r in clean_rows)),
                 mean_noise_mpiw=float(np.mean([r['MPIW'] for r in rows])),
                 mean_validation_mpiw=float(np.mean([r['MPIW'] for r in coverage_rows if r['Method']=='MANFIZ_validation'])),
                 all_optimizers_converged=all(t['success'] for t in training),
                 scope='Finite independent confirmation study; bounded-noise enclosure conditional on a valid clean envelope',
                 seconds=time.perf_counter()-started)
    write_json(output/'summary.json',summary)
    print(jsonable(summary),flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/new_training'))
    parser.add_argument('--systems',type=int,nargs='+',default=[1,2,3])
    parser.add_argument('--max-iter',type=int,default=2000)
    parser.add_argument('--max-evaluations',type=int,default=3500)
    parser.add_argument('--q-max',type=int,default=1000)
    parser.add_argument('--n-samples',type=int,default=1800)
    parser.add_argument('--calibration-replicates',type=int,default=20)
    parser.add_argument('--confirmation-replicates',type=int,default=20)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    cfg=FitConfig(premise=PremiseConfig(max_iter=args.max_iter,max_evaluations=args.max_evaluations),
                  reduction=ReductionConfig(q_max=args.q_max))
    run_benchmarks(args.output,systems=tuple(args.systems),config=cfg,n_samples=args.n_samples,
                   calibration_replicates=args.calibration_replicates,
                   confirmation_replicates=args.confirmation_replicates,plots=not args.no_plots)


if __name__ == '__main__':
    main()
