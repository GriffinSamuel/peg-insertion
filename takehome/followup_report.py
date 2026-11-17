"""Assemble the lead-written phase prose, complete scoreboard, and episode totals."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT, OUT
from takehome.followup_scoreboard import _atomic_write


def generate():
    sections=OUT/'report_sections'
    accounting=json.loads((OUT/'accounting.json').read_text())
    if accounting['status']!='valid':
        raise ValueError('Resolve accounting errors before publishing the report')
    text='# Follow-up report\n\n'+(sections/'summary.md').read_text().strip()+'\n\n'
    for phase in ('A','B','B_mixed','C','D','E','F','G'):
        path=sections/f'{phase}.md'
        if path.exists():text+=path.read_text().strip()+'\n\n'
    text+=(sections/'limits.md').read_text().strip()+'\n\n'
    counts=accounting['whole_project_episode_totals'];settings=accounting['settings']
    text+='## Episode and setting totals\n\n'
    text+=(f"Follow-up: {counts['followup_reserved_practice_episodes']:,} charged practice episodes, "
           f"{counts['followup_full_test_episodes']:,} full test-only episodes, and "
           f"{settings['new_counted_settings_from_registry']} attempted settings. "
           f"Whole project: {counts['whole_project_practice_episodes']:,} practice episodes, "
           f"{counts['whole_project_full_test_episodes']:,} full test-only episodes, "
           f"{counts['whole_project_reset_only_probes']:,} reset-only probes, and "
           f"{settings['whole_project_settings']} settings. The 49 non-learning sweep cells and "
           "the preserved partial attempt are accounted for separately from new learning configurations. "
           "[The accounting file](results/takehome/followup/accounting.json) has no validation errors.\n\n")
    scoreboard=(OUT/'scoreboard.md').read_text()
    table=scoreboard[scoreboard.index('| Phase |'):]
    text+='## Full follow-up scoreboard\n\n'+table
    for phase in ('b','b_mixed','c','d','e','f'):
        path=OUT/f'phase_{phase}/scoreboard.png'
        if path.exists():
            text+=f'\n![Phase {phase.upper()} scoreboard]({path.relative_to(ROOT).as_posix()})\n'
    _atomic_write(ROOT/'TAKEHOME_FOLLOWUP_REPORT.md',text)
    return ROOT/'TAKEHOME_FOLLOWUP_REPORT.md'


if __name__=='__main__':print(generate())
