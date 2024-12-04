from typing import Any, Dict

import flax
import jax
import numpy as np
from flashbax.buffers.trajectory_buffer import (TrajectoryBuffer,
                                                TrajectoryBufferState)

Buffer = TrajectoryBuffer
BufferState = TrajectoryBufferState
DataType = Dict[str, np.ndarray]
Params = flax.core.FrozenDict[str, Any]
PRNGKey = jax.Array
