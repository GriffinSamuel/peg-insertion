"""Measure the single locked Phase G reference pair before parallel workers."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import OUT
from takehome.followup_collect import read, save
from takehome.followup_final_worker import configurations
from takehome.followup_reference import measure_reference
import jax


def main():
    registry = read(OUT/'registry.json')
    if registry.get('status') == 'halted_tripwire':
        raise RuntimeError('Tripwire halt is active')
    if not {'A', 'B', 'B_mixed', 'C', 'D', 'E', 'F'}.issubset(registry['completed_phases']):
        raise ValueError('Finish preceding phases before fresh references')
    configs = configurations(read(OUT/'G/choice.json'))
    try:
        if not any(device.platform == 'gpu' for device in jax.devices()):
            raise RuntimeError('Tripwire: GPU not visible')
        print(measure_reference(configs[0][1]), flush=True)
    except Exception as exc:
        if 'Tripwire:' in str(exc):
            registry['status'] = 'halted_tripwire'
            registry['halted_at'] = 'G_references'
            save(OUT/'registry.json', registry)
        raise


if __name__ == '__main__':
    main()
