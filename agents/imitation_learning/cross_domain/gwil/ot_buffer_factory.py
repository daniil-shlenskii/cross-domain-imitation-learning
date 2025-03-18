from typing_extensions import override

from utils import get_buffer_state_size, instantiate_jitted_fbx_buffer
from utils.custom_types import BufferState


class OTBufferFactory:
    def __call__(self, expert_state: BufferState, batch_size: int):
        # extract buffer dict from buffer state
        expert_state_size = get_buffer_state_size(expert_state)
        expert_state_dict = {
            k: v[0, :expert_state_size] for k, v in expert_state.experience.items()
        }

        # get ot state dict
        ot_state_dict = self.get_ot_state_dict(expert_state_dict)

        # init ot buffer
        ot_state_size = ot_state_dict["observations"].shape[0]
        ot_buffer = instantiate_jitted_fbx_buffer({
            "_target_": "flashbax.make_item_buffer",
            "sample_batch_size": batch_size,
            "min_length": 1,
            "max_length": ot_state_size,
            "add_batches": False,
        })

        # init ot state
        ot_state_init_sample = {k: v[0] for k, v in ot_state_dict.items()}
        ot_state = ot_buffer.init(ot_state_init_sample)

        # fill ot state with ot dict
        ot_state_experience = {k: v[None] for k, v in ot_state_dict.items()}
        ot_state = ot_state.replace(
            experience=ot_state_experience,
            current_index=0,
            is_full=True,
        )

        return ot_buffer, ot_state 

    @override
    def get_ot_state_dict(self, expert_state_dict: dict):
        return expert_state_dict


class ReverseOTBufferFactory(OTBufferFactory): 
    def get_ot_state_dict(self, expert_state_dict: dict):
        expert_state_dict["observations"], expert_state_dict["observations_next"] =\
            expert_state_dict["observations_next"], expert_state_dict["observations"]
        return expert_state_dict
