from typing import Any

import flax

from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as Buffer


DataType = "DataType"
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any