from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv


class CustomPointUmaze(PointMazeEnv):
    def reset(self, *args, seed=None, **kwargs):
        options = {"goal_cell": (1, 1), "reset_cell": (3, 1)}
        return super().reset(seed=seed, options=options)

class CustomPointUmazeInverse(PointMazeEnv):
    def reset(self, *args, seed=None, **kwargs):
        options = {"goal_cell": (3, 1), "reset_cell": (1, 1)}
        return super().reset(seed=seed, options=options)
