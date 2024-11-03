from typing import Any

import flax

from flashbax.buffers.trajectory_buffer import TrajectoryBuffer, TrajectoryBufferState


DataType = "DataType"
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Buffer = TrajectoryBuffer
BufferState = TrajectoryBufferState