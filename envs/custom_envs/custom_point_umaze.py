from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv


class CustomPointUmaze(PointMazeEnv):
    def __init__(
        self,
        reward_type="dense",
        render_mode=None,
    ):
        super().__init__(
            reward_type=reward_type,
            render_mode=render_mode,
            continuing_task=True,
            reset_target=False,
        )


    def reset(self, *args, seed=None, **kwargs):
        options = {"goal_cell": (1, 1), "reset_cell": (3, 1)}
        return super().reset(seed=seed, options=options)
