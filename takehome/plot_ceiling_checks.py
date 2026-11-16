"""Plot recorded independent ceiling checks for physical inspection only."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / 'results/takehome'


def main():
    folder = OUT / 'ceiling_reviews'
    folder.mkdir(exist_ok=True)
    rows=[]
    paths=[]
    for phase in ('goal7','goal8','goal9','goal10','goal11','goal13'):
        paths.extend((OUT/phase).glob('**/extra_ceiling*.json'))
    for path in sorted(paths):
        data=json.loads(path.read_text())
        if not data.get('test_only'):
            continue
        assert data['test_seed']==789 and data['episodes']==500
        assert data['learned_from'] is False and data['used_for_selection'] is False
        candidates=[path.with_suffix('.npz'),path.with_name(path.stem+'_traces.npz')]
        trace_path=next(p for p in candidates if p.exists())
        with np.load(trace_path) as archive:
            assert bool(archive['test_only']) and not bool(archive['learned_from'])
            assert int(archive['test_seed'])==789
            ids=data['inspected_episode_ids'][:3]
            depth=archive['depth'][:,ids]*1000
            radius=archive['xy_err'][:,ids]*1000
            action=np.max(np.abs(archive['executed_action'][:,ids]),axis=-1)
        assert all(np.isfinite(v).all() for v in (depth,radius,action))
        assert np.max(action)<=1.00001
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
        for j,episode in enumerate(ids):
            for ax,values in zip(axes,(depth,radius,action)):
                ax.plot(np.arange(len(values)),values[:,j],label=f'Episode {episode}')
        for ax,label in zip(axes,('Insertion depth (mm)','True radial XY error (mm)','Max |command| (normalized)')):
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[-1].set_xlabel('Recorded control step')
        axes[-1].axhline(1,color='gray',linestyle=':',label='Command bound')
        axes[0].legend(ncol=3)
        fig.suptitle(str(path.relative_to(OUT))+'\nIndependent test seed 789; not used to select or replace a score',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.94))
        output=folder/('_'.join(path.relative_to(OUT).with_suffix('').parts)+'.png')
        fig.savefig(output,dpi=150);plt.close(fig)
        rows.append(dict(source=str(path.relative_to(ROOT)),trace=str(trace_path.relative_to(ROOT)),
            chart=str(output.relative_to(ROOT)),episode_ids=ids,
            successes=data['successes'],test_episodes=500,test_seed=789,
            through_wall_episodes=data['through_wall_episodes'],nan_episodes=data['nan_episodes'],
            max_lateral_when_deep_mm=data['max_lateral_when_deep_mm'],
            example_max_depth_mm=np.max(depth,axis=0).tolist(),
            example_max_abs_action=np.max(action,axis=0).tolist(),
            used_for_selection=False,visual_review_required=True))
    (folder/'summary.json').write_text(json.dumps(dict(checks=rows,new_episodes=0),indent=2)+'\n')
    print(json.dumps(dict(plotted_checks=len(rows),new_episodes=0)))


if __name__=='__main__':
    main()
