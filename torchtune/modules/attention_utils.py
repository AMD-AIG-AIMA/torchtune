# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from typing import Callable, List, Optional, Union

import torch

from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION
from torchtune.utils._logging import get_logger, log_once
from typing import Tuple

_log: logging.Logger = get_logger()


if os.environ.get("FLASH_ATTN_AVAILABLE", True):
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
    except ModuleNotFoundError:
        log_once(_log, "Not using flash_attn.", level=logging.DEBUG)
        flash_attn_func = None
else:
    flash_attn_func = None


def is_flash_attn_available():
    return flash_attn_func is not None


if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask as create_block_causal_mask_flex,
        flex_attention,
    )

    flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

    # We cannot do nested compile, but flex attention only has perf benefits
    # when compiled. To insulate it from the compiler, we wrap it with
    # compiler.disable so that it can be used regardless of whether the model
    # is compiled or not, and flex attention always remains compiled.
    @torch.compiler.disable(recursive=False)
    def compile_friendly_flex_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        return flex_attention_compiled(q, k, v, block_mask=block_mask)

    _MaskType = Union[torch.Tensor, BlockMask]
else:
    _MaskType = torch.Tensor


def _check(attn_mask: torch.Tensor) -> torch.bool:
    """
    Checks if attn_mask is compatible with flash_attn_wrapper.

    Args:
        attn_mask (torch.Tensor): attention mask of shape
            (batch_size, max_q_seqlen, max_kv_seqlen)

    Returns:
        is_invalid (torch.bool): True if incompatible with flash_attn_wrapper
    """
    n_rows, n_cols = attn_mask.shape[-2:]
    n_null_rows = (~attn_mask).all(dim=-1).sum(dim=-1)
    n_null_cols = (~attn_mask).all(dim=-2).sum(dim=-1)
    is_invalid = (
        attn_mask.numel() - (
            n_cols * n_null_rows.sum() +
            n_rows * n_null_cols.sum() -
            (n_null_rows * n_null_cols).sum()
        )
    ) > attn_mask.sum()

    return is_invalid


def _unpad_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    """
    batch_size, q_max_seqlen, k_max_seqlen = attn_mask.shape
    q_mask = attn_mask.any(dim=2) #sum(dim=2).to(torch.bool)
    kv_mask = attn_mask.any(dim=1) #sum(dim=1).to(torch.bool)
    q_flat, q_indices, q_cu_seqlens, q_max_seqlen, q_used_seqlens  = unpad_input(q, q_mask)
    k_flat, k_indices, k_cu_seqlens, k_max_seqlen, k_used_seqlens  = unpad_input(k, kv_mask)
    v_flat, v_indices, v_cu_seqlens, v_max_seqlen, v_used_seqlens  = unpad_input(v, kv_mask)
    return (
        q_flat, k_flat, v_flat,
        (q_indices, q_cu_seqlens, q_max_seqlen),
        (k_indices, k_cu_seqlens, k_max_seqlen),
    )


# converts tensors from sdpa format to fa format
def _flash_attn_wrapper(q, k, v, attn_mask, dropout_p, is_causal):
    q=q.transpose(1, 2) # Transpose to (N, L, H, E)
    k=k.transpose(1, 2) # Transpose to (N, L, H, E)
    v=v.transpose(1, 2) # Transpose to (N, L, H, E)
    if attn_mask is not None and attn_mask.numel() > attn_mask.sum():
       if _check(attn_mask):
            raise ValueError("invalid 'attn_mask' found")

        batch_size, q_len  = q.shape[:2]
        query_states, key_states, value_states, mask_info_q, mask_info_kv  = _unpad_inputs(
            q, k, v, attn_mask,
        )
        indices_q, cu_seqlens_q, max_seqlen_q = mask_info_q
        indices_k, cu_seqlens_k, max_seqlen_k = mask_info_kv
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            causal=is_causal,
        )
        out = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    else:
        out = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=is_causal,
        )
    return out.transpose(1, 2)  # Transpose back to (N, H, L, E)


def _get_document_ids_from_seq_lens(
    seq_lens: List[torch.Tensor],
) -> torch.Tensor:
    """
    Convert a batch tensor of seq lens into integer IDs denoting sample ownership.
    For example, seq_lens = [2, 3, 1] would return [0, 0, 1, 1, 1, 2].

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        Tensor: Document IDs of shape (batch_size, max_seq_len).
    """
    batch_size = len(seq_lens)
    batch_document_ids = []
    for sample_idx in range(batch_size):
        # We assume seq lens sum to max seq lens, so document_ids should be of
        # shape (max_seq_len, )
        document_ids = torch.cat(
            [
                torch.full((seq_len,), i, dtype=torch.long, device=seq_len.device)
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    batch_document_ids = torch.stack(batch_document_ids)
    return batch_document_ids


def create_block_causal_mask(seq_lens: List[torch.Tensor]) -> torch.Tensor:
    """
    Given a batch tensor of seq lens defining the lengths of samples in each pack,
    Construct a 2D block causal mask for each pack in the batch. For example, if
    a single sample's seq_lens is [3, 2, 1], the mask would be::

        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.


    Returns:
        Tensor: Block causal mask of shape (batch_size, max_seq_len, max_seq_len).
    """
    batch_block_attn_masks = []
    batch_size = len(seq_lens)
    for sample_idx in range(batch_size):
        block_attn_masks = [
            torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq_len.device)
            )
            for i, seq_len in enumerate(seq_lens[sample_idx])
        ]

        batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))
    return torch.stack(batch_block_attn_masks)


def packed_block_causal_mask(
    seq_lens: List[torch.Tensor],
) -> _MaskType:
    """
    Create a block causal document mask for a batch of packed sequences. If
    flex attention is supported by the current hardware, block causal logic and
    passing this into :func:`torch.nn.attention.flex_attention.create_block_mask`.
    The resultant BlockMask is a compressed representation of the full block causal
    mask. If on an older version, a standard 2D block causal mask is created and returned.

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        _MaskType: BlockMask or Tensor if torch version < 2.5.0.
    """
    if _SUPPORTS_FLEX_ATTENTION:
        document_ids = _get_document_ids_from_seq_lens(seq_lens)
        batch_size, max_seq_len = document_ids.shape
        document_ids = document_ids.to("cuda")

        # Instead of passing a tensor mask, flex attention requires a mask_mod function
        # that determines which elements of QK^T should be included in the attention
        # computation prior to the softmax. For sample packing, we need both the
        # logic for both causal mask and document mask. See PyTorch's official
        # blog post for more details: https://pytorch.org/blog/flexattention/#mask-mods
        def mask_mod(b, h, q_idx, kv_idx):
            """
            Defines the logic of a block causal mask by combining both a standard causal mask
            and a block diagonal document mask.

            See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
            for an illustration.
            """
            causal_mask = q_idx >= kv_idx
            document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
            return causal_mask & document_mask

        return create_block_causal_mask_flex(
            mask_mod,
            batch_size,
            None,
            max_seq_len,
            max_seq_len,
            device="cuda",
        )
    else:
        return create_block_causal_mask(seq_lens=seq_lens)


def _sdpa_or_flex_attention() -> Callable:
    """
    Helper function to decide when to call flex attention or SDPA. It will use
    flex attention if ALL of the following conditions are met, otherwise it will
    default to SDPA:
    - torch version >= 2.5.0
    - we are sample packing, therefore mask is a BlockMask
    - torch.cuda.get_device_capability() >= (7, 5)
    """

    if _SUPPORTS_FLEX_ATTENTION:

        def _attention_call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[_MaskType],
            dropout_p: float,
            is_causal: bool,
            with_kv_cache: bool,
        ) -> torch.Tensor:

            # Flex attention uses the BlockMask
            # (https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L168)
            # instead of a traditional boolean tensor mask. If this is passed in,
            # we assume the user wants to use flex attention instead of traditional SDPA.
            # This will use flash attention under the hood with support for custom masks.
            # Currently, it is used when sample packing is enabled (see torchtune.datasets.PackedDataset)
            if isinstance(mask, BlockMask):
                log_once(
                    _log,
                    "Using flex attention for attention computation since a BlockMask was passed in.",
                    level=logging.DEBUG,
                )
                if dropout_p > 0.0:
                    raise ValueError(
                        "Flex attention does not support dropout. Please set dropout to 0.0."
                    )
                return compile_friendly_flex_attention(
                    q,
                    k,
                    v,
                    block_mask=mask,
                )
            # If mask is a standard boolean tensor or None, then use SDPA
            else:
                if is_flash_attn_available() and not with_kv_cache:
                    log_once(_log, "Using flash_attn", level=logging.DEBUG)
                    try:
                        output = _flash_attn_wrapper(
                            q,
                            k,
                            v,
                            attn_mask=mask,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                        )
                    except ValueError:
                        log_once(
                            _log,
                            (
                                "Invalid 'attn_mask' in flash_attn_wrapper, resorting "
                                "to 'scaled_dot_product_attention'"
                            ),
                            level=logging.DEBUG
                        )
                        mask = mask[:, None, :, :]

                        try:
                            output = nn.functional.scaled_dot_product_attention(
                                q,
                                k,
                                v,
                                attn_mask=mask,
                                dropout_p=dropout_p,
                                is_causal=is_causal,
                            )
                        except Exception:
                            log_once(
                                _log,
                                "Using SDPBackend.MATH",
                                level=logging.DEBUG
                            )
                            with sdpa_kernel(SDPBackend.MATH):
                                output = nn.functional.scaled_dot_product_attention(
                                    q,
                                    k,
                                    v,
                                    attn_mask=mask,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                )
                    return output
                else:
                    # shape: [b, 1, s, s]
                    if mask is not None:
                        mask = mask[:, None, :, :]

                    # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
                    return nn.functional.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                    )

    else:
        def _attention_call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[_MaskType],
            dropout_p: float,
            is_causal: bool,
            with_kv_cache: bool,
        ) -> torch.Tensor:
            if is_flash_attn_available() and not with_kv_cache:
                log_once(_log, "Using flash_attn", level=logging.DEBUG)
                try:
                    output = _flash_attn_wrapper(
                        q,
                        k,
                        v,
                        attn_mask=mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                    )
                except ValueError:
                    log_once(
                        _log,
                        (
                            "Invalid 'attn_mask' in flash_attn_wrapper, resorting "
                                "to 'scaled_dot_product_attention'"
                        ),
                        level=logging.DEBUG
                    )
                    mask = mask[:, None, :, :]

                    try:
                        output = nn.functional.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=mask,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                        )
                    except Exception:
                        log_once(
                            _log,
                            "Using SDPBackend.MATH",
                            level=logging.DEBUG
                        )
                        with sdpa_kernel(SDPBackend.MATH):
                            output = nn.functional.scaled_dot_product_attention(
                                q,
                                k,
                                v,
                                attn_mask=mask,
                                dropout_p=dropout_p,
                                is_causal=is_causal,
                            )
                return output
            else:
                # shape: [b, 1, s, s]
                if mask is not None:
                    mask = mask[:, None, :, :]

                # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
                return nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )

    return _attention_call
