"""Review saved required above-reference diagnostics, without new episodes."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import OUT
from takehome.followup_scoreboard import _atomic_write
import numpy as np


def review(phase):
    folder=OUT/phase
    rows=[];example=None
    for path in sorted(folder.glob('**/extra_ceiling_[0-9][0-9][0-9][0-9].json')):
        row=json.loads(path.read_text())
        if (row['episodes']!=500 or row['test_seed']!=789 or row['learned_from']
                or not row['test_only'] or row['used_for_selection']):
            raise ValueError(f'Invalid physical-check provenance: {path}')
        if row['through_wall_episodes'] or row['nan_episodes']:
            raise ValueError(f'Physical check requires investigation: {path}')
        controls=np.asarray(row['max_abs_control_examples'])
        if not np.isfinite(controls).all() or np.any(controls>1.000001):
            raise ValueError(f'Invalid inspected control: {path}')
        trace_path=path.with_name(path.stem+'_traces.npz')
        with np.load(trace_path) as trace:
            if (int(trace['test_seed'])!=789 or not bool(trace['test_only'])
                    or bool(trace['learned_from'])
                    or int(np.sum(trace['episode_success']>.5))!=row['successes']):
                raise ValueError(f'Physical trace does not match summary: {path}')
            if example is None:
                example=dict(path=str(path.relative_to(OUT)),
                    ids=row['inspected_episode_ids'],
                    tip_height=trace['tip_height'][:,row['inspected_episode_ids']],
                    xy_err=trace['xy_err'][:,row['inspected_episode_ids']],
                    force=trace['force'][:,row['inspected_episode_ids']])
        rows.append(dict(path=str(path.relative_to(OUT)),episodes=500,test_seed=789,
            successes=row['successes'],through_wall_episodes=row['through_wall_episodes'],
            nan_episodes=row['nan_episodes'],inspected=row['inspected_episode_ids'],
            max_lateral_when_deep_examples_m=row['max_lateral_when_deep_examples_m'],
            max_abs_control_examples=row['max_abs_control_examples'],used_for_selection=False))
    report=dict(phase=phase,source='Read-only review of required diagnostic tests; no new episodes.',
        checks=len(rows),test_episodes=500*len(rows),inspected_examples=sum(len(r['inspected']) for r in rows),
        through_wall_episodes=0,nan_episodes=0,used_for_selection=False,rows=rows)
    if example is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for axis,key,title,scale in zip(axes,('tip_height','xy_err','force'),
                ('Tip height relative to opening (mm)','True sideways distance (mm)','Contact force (N)'),
                (1000,1000,1)):
            axis.plot(np.arange(1,201),example[key]*scale)
            axis.set(xlabel='Control step',ylabel=title)
            axis.grid(alpha=.2)
        axes[0].axhline(0,color='black',linestyle=':',linewidth=1)
        axes[0].legend([f'Episode {i}' for i in example['ids']],fontsize=7)
        fig.suptitle('Saved physical-check examples: '+phase)
        fig.tight_layout()
        fig.savefig(folder/'physical_examples.png',dpi=160)
        plt.close(fig)
        report['example_plot_source']=example['path']
    _atomic_write(folder/'physical_review.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
    return {k:v for k,v in report.items() if k!='rows'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('B_mixed','C','D','E','G'))
    print(json.dumps(review(p.parse_args().phase)))
