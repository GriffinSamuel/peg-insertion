"""Stable identities for camera contexts and preserved/restarted experiments."""
from takehome.followup_runtime import OUT


def condition_key(config):
    if config.camera_condition=='mixed':
        return f'mixed_fixed{round(config.mixed_fixed_std*1e6)}_episode{round(config.mixed_episode_std*1e6)}'
    return config.camera_condition


def run_path(entry,seed):
    custom=entry.get('seed_paths',{}).get(str(seed))
    path=OUT/custom if custom else OUT/entry['phase']/entry['name']/f'seed_{seed}'
    path=path.resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('Registered output must remain inside followup')
    return path


def entry_condition_key(entry):
    if entry['condition']=='mixed':
        values=entry['overrides']
        return f"mixed_fixed{round(values['mixed_fixed_std']*1e6)}_episode{round(values['mixed_episode_std']*1e6)}"
    return entry['condition']
