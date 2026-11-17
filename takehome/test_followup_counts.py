"""Report repeated controls honestly without changing their run budgets."""
import pytest
from takehome.followup_accounting import setting_counts
from takehome.followup_e_manifest import required_settings


def test_required_e_has_six_attempts_but_three_new_distinct_settings():
    rows=required_settings()
    for row in rows:row['counted']=True
    counts=setting_counts(rows,28)
    assert counts['new_counted_settings_from_registry']==6
    assert counts['new_distinct_settings']==3
    assert counts['whole_project_distinct_settings']==31
    assert counts['whole_project_settings']==34
    assert counts['repeated_original_setting_count']==3
    assert setting_counts(required_settings(),28)['repeated_original_setting_count']==0


def test_changed_config_cannot_be_deducted_as_an_original_repeat():
    row=required_settings()[0];row['counted']=True
    row['overrides']['ensemble']=1
    with pytest.raises(ValueError,match='unchanged normal Phase E'):
        setting_counts([row],28)
