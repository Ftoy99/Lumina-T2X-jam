from typing import List, Tuple

import torch


@staticmethod
def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        scale_watershed: float = 1.0,
        timestep: float = 1.0,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with
    given dimensions.

    This function calculates a frequency tensor with complex exponentials
    using the given dimension 'dim' and the end index 'end'. The 'theta'
    parameter scales the frequencies. The returned tensor contains complex
    values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation.
            Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex
            exponentials.
    """

    if timestep < scale_watershed:
        linear_factor = scale_factor
        ntk_factor = 1.0
    else:
        linear_factor = 1.0
        ntk_factor = scale_factor

    theta = theta * ntk_factor
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) / linear_factor

    timestep = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore

    freqs = torch.outer(timestep, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    freqs_cis_h = freqs_cis.view(end, 1, dim // 4, 1).repeat(1, end, 1, 1)
    freqs_cis_w = freqs_cis.view(1, end, dim // 4, 1).repeat(end, 1, 1, 1)
    freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1).flatten(2)

    return freqs_cis


freqs_cis = precompute_freqs_cis(
    64 // 32,
    384,
    scale_factor=1,
)


def patchify_and_embed(
        self, x: List[torch.Tensor] | torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
    global freqs_cis
    freqs_cis = freqs_cis.to(x[0].device)
    if isinstance(x, torch.Tensor):
        pH = pW = 8
        B, C, H, W = x.size()
        x = x.view(B, C, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(3)

        # x = self.x_cat_emb(x)

        x = x.flatten(1, 2)
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)

        return (
            x,
            mask,
            [(H, W)] * B,
            freqs_cis[: H // pH, : W // pW].flatten(0, 1).unsqueeze(0),
        )
    else:
        pH = pW = self.patch_size
        x_embed = []
        freqs_cis = []
        img_size = []
        l_effective_seq_len = []

        for img in x:
            C, H, W = img.size()
            item_freqs_cis = self.freqs_cis[: H // pH, : W // pW]
            freqs_cis.append(item_freqs_cis.flatten(0, 1))
            img_size.append((H, W))
            img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
            img = self.x_cat_emb(img)
            img = img.flatten(0, 1)
            l_effective_seq_len.append(len(img))
            x_embed.append(img)

        max_seq_len = max(l_effective_seq_len)
        mask = torch.zeros(len(x), max_seq_len, dtype=torch.int32, device=x[0].device)
        padded_x_embed = []
        padded_freqs_cis = []
        for i, (item_embed, item_freqs_cis, item_seq_len) in enumerate(
                zip(x_embed, freqs_cis, l_effective_seq_len)
        ):
            item_embed = torch.cat(
                [
                    item_embed,
                    self.pad_token.view(1, -1).expand(max_seq_len - item_seq_len, -1),
                ],
                dim=0,
            )
            item_freqs_cis = torch.cat(
                [
                    item_freqs_cis,
                    item_freqs_cis[-1:].expand(max_seq_len - item_seq_len, -1),
                ],
                dim=0,
            )
            padded_x_embed.append(item_embed)
            padded_freqs_cis.append(item_freqs_cis)
            mask[i][:item_seq_len] = 1

        x_embed = torch.stack(padded_x_embed, dim=0)
        freqs_cis = torch.stack(padded_freqs_cis, dim=0)
        return x_embed, mask, img_size, freqs_cis


if __name__ == '__main__':
    x = torch.randn([1, 4, 64, 64])
    xmf = torch.randn([1, 4, 64, 64])
    patches = patchify_and_embed(x, xmf)
    print(f"Patch shape {patches.shape}")
