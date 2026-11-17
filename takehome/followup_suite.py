"""Four detached GPU jobs with one parent writing the scoreboard."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT,OUT
from takehome.followup_collect import collect,read,save
import argparse,subprocess,time,json


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--settings',nargs='+',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=None);a=p.parse_args()
    if len(set(a.settings))!=len(a.settings) or (a.seeds is not None and len(set(a.seeds))!=len(a.seeds)):
        raise ValueError('Duplicate settings or seeds would risk concurrent writes')
    registry=read(OUT/'registry.json')
    if registry.get('status')=='halted_tripwire':
        raise RuntimeError('Tripwire halt is active; user direction is required before any new job')
    lookup={e['name']:e for e in registry['settings']}
    requested={}
    for name in a.settings:
        entry=lookup[name]
        requested[name]=a.seeds if a.seeds is not None else entry['seeds']
        if entry['status'] not in ('planned','running','finished'):
            raise ValueError('A skipped/failed setting needs explicit lead review before launch')
        if not set(requested[name]).issubset(set(entry['seeds'])):raise ValueError('Seeds not registered')
    will_run=lambda e: bool(e.get('counted')) or e['name'] in a.settings
    if sum(will_run(e) for e in lookup.values())>90:raise ValueError('90-setting cap')
    if sum(will_run(e) and e['phase']=='B' and e['overrides'].get('outputs',2)!=3 for e in lookup.values())>14:
        raise ValueError('14-aim-variant cap')
    save(OUT/'registry.json',registry)
    todo=list(a.settings);active=[];done=[];tripwire=False;started=time.perf_counter()
    while todo or active:
        while todo and len(active)<4 and not tripwire:
            name=todo.pop(0);log_path=OUT/'logs'/f'{a.name}_{name}.log';log_path.parent.mkdir(exist_ok=True)
            log=log_path.open('x')
            lookup[name]['status']='running';lookup[name]['counted']=True
            save(OUT/'registry.json',registry)
            process=subprocess.Popen([sys.executable,'-B','takehome/followup_job.py','--setting',name,'--seeds',
                *map(str,requested[name])],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active.append((name,process,log,log_path,time.perf_counter()))
            print('START',name,process.pid,'SEEDS',requested[name],flush=True)
        for item in list(active):
            name,process,log,path,t0=item;code=process.poll()
            if code is None:continue
            log.close();active.remove(item)
            row=dict(name=name,exit_code=code,seconds=time.perf_counter()-t0);done.append(row)
            print('FINISH',json.dumps(row),flush=True)
            if code:
                text=path.read_text();tripwire=tripwire or 'Tripwire:' in text
                if 'Tripwire:' in text:
                    registry['status']='halted_tripwire';registry['halted_at']=name
                lookup[name]['status']='failed';lookup[name]['reason']=f'Job exit{code}; see {path.relative_to(OUT)}'
            else:lookup[name]['status']='finished'
            save(OUT/'registry.json',registry)
            try:collect()
            except Exception as exc:
                tripwire=True
                if 'Tripwire:' in str(exc):
                    registry['status']='halted_tripwire';registry['halted_at']=name
                    save(OUT/'registry.json',registry)
                print('COLLECTION FAILURE',repr(exc),flush=True)
                save(OUT/f'{a.name}_collection_failure.json',dict(error=repr(exc),setting=name))
            if tripwire:
                todo.clear()
                for _,other,_,_,_ in active:other.terminate()
        if active:time.sleep(1)
    save(OUT/f'{a.name}_suite.json',dict(jobs=done,tripwire=tripwire,seconds=time.perf_counter()-started))
    raise SystemExit(1 if tripwire or any(r['exit_code'] for r in done) else 0)

if __name__=='__main__':main()
