import math
from abc import ABC, abstractmethod
import torch


class CustomRoPE:
    @staticmethod
    def pad_position_ids(position_ids: list, padding_mode="right", padding_value=0):
        """
        position_ids shape: [seq_len, dim]
        """
        max_len = max(pos.shape[0] for pos in position_ids)
        padded_position_ids = []
        for pos in position_ids:
            padding_tensor = torch.zeros((max_len - pos.shape[0], pos.shape[1]), dtype=pos.dtype, device=pos.device) + padding_value
            if padding_mode == "right":
                pad_list = [pos, padding_tensor]
            else:
                pad_list = [padding_tensor, pos]
            padded_position_ids.append(torch.cat(pad_list, dim=0))
        return torch.stack(padded_position_ids, dim=0)


class DataShape(ABC):
    @abstractmethod
    def __call__(self, rope_obj: CustomRoPE) -> torch.Tensor:
        pass


class VRoPEVideoShape(DataShape):
    def __init__(self, t: int, h: int, w: int):
        self.t = t
        self.h = h
        self.w = w
    
    def __call__(self, rope_obj: "VRoPE") -> torch.Tensor:
        return rope_obj.generate_nd_positions((self.t, self.h, self.w))


class VRoPEImageShape(VRoPEVideoShape):
    def __init__(self, h: int, w: int):
        super().__init__(1, h, w)


class VRoPETextShape(DataShape):
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
    
    def __call__(self, rope_obj: "VRoPE") -> torch.Tensor:
        return rope_obj.generate_1d_positions(self.seq_len)


class VRoPESharedTextShape(DataShape):
    # NOTE: VRoPE shared text shape allows generating grouped suffix input

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
    
    def __call__(self, rope_obj: "VRoPE") -> torch.Tensor:
        # NOTE: do not increase ``current_index``
        position_ids = torch.arange(rope_obj.current_index, rope_obj.current_index + self.seq_len, dtype=torch.long).unsqueeze(-1).expand(-1, rope_obj.half_head_dim)
        rope_obj.total_token_len += self.seq_len
        return position_ids


class VRoPE(CustomRoPE):
    """
    VRoPE implementation for any-dimensional input.
    """

    def __init__(
        self,
        half_head_dim: int,
        half_time_dim: int,
    ):
        assert half_time_dim < half_head_dim, "``half_time_dim`` must be smaller than ``half_head_dim``"
        # channel allocation config
        self.half_head_dim = half_head_dim
        # self.half_time_dim = half_time_dim
        self.half_time_dim = 0
        # total length of the tokens
        self.total_token_len = 0
        # current position index
        self.current_index = 0
        # interleaved data shapes
        self.data_shapes: list[DataShape] = []

    def __call__(self):
        if not self.data_shapes:
            # Assume that it's autoregressive mode
            return self.generate_1d_positions(1)

        position_ids = torch.cat([data_shape(self) for data_shape in self.data_shapes], dim=0)
        # Clear after data shapes are computed
        self.data_shapes.clear()
        return position_ids

    def generate_1d_positions(self, seq_len: int):
        """
        Generate 1d positions given seq_len.
        """
        position_ids = torch.arange(self.current_index, self.current_index + seq_len, dtype=torch.long).unsqueeze(-1).expand(-1, self.half_head_dim)
        self.increase_index_and_total_len(seq_len, seq_len)
        return position_ids  # seq_len, half_head_dim

    def generate_nd_positions(self, data_shape: tuple):
        """
        Generate nd positions given (time, d1, d2, d3, ..., dn).
        TODO: Add mixed nd positions (e.g., cls + patch)
        """
        time_shape = data_shape[0]
        nd_shape = data_shape[1:]
        n = len(nd_shape)
        nd_len = math.prod(nd_shape)  # number of tokens in nd
        nd_index_len = sum(nd_shape) - n + 1  # number of indexes used in nd

        def generate_xd_positions(x):
            """
            Generate positions at xd (x >= 0).
            """
            shape = nd_shape[x]
            shape_before = math.prod(nd_shape[0:x])
            shape_after = math.prod(nd_shape[x + 1 :])
            positions = torch.arange(shape, dtype=torch.long).unsqueeze(0).unsqueeze(-1).expand(shape_before, -1, shape_after)
            # set prod_shape[x] = 2 to apply broadcasting in the following sum operation.
            prod_shape = [1] * n
            prod_shape[x] = 2
            return torch.stack(
                [
                    positions.flatten(),
                    positions.flip(dims=(1,)).flatten(),
                ],
                dim=0,
            ).view(*(prod_shape + [nd_len]))

        nd_positions = list(map(generate_xd_positions, range(len(nd_shape))))
        # Apply symmetric operations.
        two_pow_nd_positions = sum(nd_positions).reshape(-1, nd_len).unsqueeze(0).expand(time_shape, -1, -1)
        # Add time shift.
        # TODO: add time delta
        time_positions = (torch.arange(time_shape, dtype=torch.long) * nd_index_len).unsqueeze(-1).expand(-1, nd_len)
        two_pow_nd_positions = two_pow_nd_positions + time_positions.unsqueeze(1)
        # Make time index center
        time_positions = time_positions + nd_index_len // 2
        # flatten to 1d sequence and add ``current_index`` offset.
        time_positions = time_positions.flatten() + self.current_index  # seq_len
        two_pow_nd_positions = two_pow_nd_positions.permute(1, 0, 2).flatten(start_dim=1) + self.current_index  # 2^n, seq_len
        # channel allocation
        # TODO: channel allocation strategies
        two_pow_nd_position_ids = torch.stack([two_pow_nd_positions[i % two_pow_nd_positions.shape[0]] for i in range(self.half_head_dim - self.half_time_dim)], dim=-1)  # seq_len, half_head_dim - half_time_dim
        time_position_ids = time_positions.unsqueeze(-1).expand(-1, self.half_time_dim)  # seq_len, half_time_dim
        position_ids = torch.cat([two_pow_nd_position_ids, time_position_ids], dim=-1)  # seq_len, half_head_dim
        self.increase_index_and_total_len(nd_index_len * time_shape, nd_len * time_shape)
        return position_ids  # seq_len, half_head_dim

    def add_shape(self, shape: DataShape) -> None:
        """
        Add a data shape to ``self.data_shapes``.
        """
        self.data_shapes.append(shape)

    def increase_index_and_total_len(self, index_len: int, token_len: int) -> None:
        """
        The ``current_index`` and ``total_token_len`` should always be consistent, so we need to increase them together.
        """
        self.current_index += int(index_len)
        self.total_token_len += int(token_len)


class GroupedRoPE(CustomRoPE):
    def __call__(self, group_info: list, padding_mode: str):
        """
        Apply 1d RoPE based on group_info
        """
        position_ids = []
        for group in group_info:
            prefix_len = group[0]
            grouped_position_ids = [torch.arange(prefix_len)]
            for g in group[1:]:
                grouped_position_ids.append(torch.arange(prefix_len, prefix_len + g))
            position_ids.append(torch.cat(grouped_position_ids, dim=0).unsqueeze(-1))
        return self.pad_position_ids(position_ids=position_ids, padding_mode=padding_mode).squeeze(-1)
