from abc import ABC, abstractmethod
from typing import Dict, Tuple

from flax import struct
from flax.struct import PyTreeNode

from utils import SaveLoadFrozenDataclassMixin
from utils.types import DataType


class BaseDomainEncoderDiscriminators(PyTreeNode, SaveLoadFrozenDataclassMixin, ABC):
    has_state_discriminator_paired_input: str = struct.field(pytree_node=False)
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @abstractmethod
    def get_state_discriminator(self):
        pass

    @abstractmethod
    def get_policy_discriminator(self):
        pass

    @abstractmethod
    def get_state_loss(self):
        pass

    @abstractmethod
    def get_policy_loss(self):
        pass

    @abstractmethod
    def update(
        self,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ) -> Tuple["BaseDomainEncoderDiscriminators", Dict, Dict]:
        pass
