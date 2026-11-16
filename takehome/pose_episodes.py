"""Record Goal 9 simulator poses for rendering without replaying physics."""
from takehome.episodes import EpisodeRunner, TRACE_KEYS


class PoseEpisodeRunner(EpisodeRunner):
    trace_keys = TRACE_KEYS + ('qpos','mocap_pos','pose_succeeded','true_rel')

    def __init__(self,*args,**kwargs):
        if not hasattr(EpisodeRunner,'extra_trace'):
            raise RuntimeError('Install the analysis trace hook after Goal7 training before pose recording')
        super().__init__(*args,**kwargs)

    def extra_trace(self, state, nxt):
        return dict(qpos=nxt.data.qpos,mocap_pos=nxt.data.mocap_pos,
                    pose_succeeded=nxt.info['succeeded'],true_rel=(
                        nxt.data.site_xpos[:,self.env._tip_site]-nxt.data.site_xpos[:,self.env._entrance_site]))
