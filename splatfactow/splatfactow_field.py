"""Fields for Splatfacto-W.

Forward-looking but backwards-compatible with the Nerfstudio 1.1.5 / gsplat 1.4.x
stack. The main goals here are:
- avoid unnecessary tensor repeats where possible,
- expose stable helper methods for SH export / rendering,
- keep call sites simple for later gsplat upgrades.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor, nn

from gsplat.cuda._wrapper import spherical_harmonics
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components import MLP
from nerfstudio.fields.base_field import Field


class BGField(Field):
    def __init__(
        self,
        appearance_embedding_dim: int,
        implementation: Literal["tcnn", "torch"] = "torch",
        sh_levels: int = 4,
        layer_width: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.sh_dim = (sh_levels + 1) ** 2

        self.encoder = MLP(
            in_dim=appearance_embedding_dim,
            num_layers=num_layers - 1,
            layer_width=layer_width,
            out_dim=layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation=implementation,
        )
        self.sh_base_head = nn.Linear(layer_width, 3)
        self.sh_rest_head = nn.Linear(layer_width, (self.sh_dim - 1) * 3)
        self.sh_rest_head.weight.data.zero_()
        self.sh_rest_head.bias.data.zero_()

    def _encode(self, appearance_embedding: Optional[Tensor]) -> Tensor:
        if appearance_embedding is None:
            raise ValueError("appearance_embedding must be provided for BGField")
        if appearance_embedding.dim() == 1:
            appearance_embedding = appearance_embedding.unsqueeze(0)
        return self.encoder(appearance_embedding).float()

    def get_sh_coeffs(
        self,
        appearance_embedding: Optional[Tensor] = None,
        num_sh: Optional[int] = None,
    ) -> Tensor:
        x = self._encode(appearance_embedding)
        full_sh_dim = self.sh_dim
        target_sh_dim = full_sh_dim if num_sh is None else (num_sh + 1) ** 2
        target_sh_dim = min(target_sh_dim, full_sh_dim)

        base_color = self.sh_base_head(x)
        sh_rest = self.sh_rest_head(x)[..., : max(target_sh_dim - 1, 0) * 3]
        return torch.cat([base_color, sh_rest], dim=-1).view(-1, target_sh_dim, 3)

    def get_background_rgb(
        self,
        ray_bundle: RayBundle,
        appearance_embedding: Optional[Tensor] = None,
        num_sh: int = 4,
    ) -> Tensor:
        directions = ray_bundle.directions.reshape(-1, 3)
        sh_coeffs = self.get_sh_coeffs(appearance_embedding=appearance_embedding, num_sh=num_sh)
        if sh_coeffs.shape[0] == 1 and directions.shape[0] != 1:
            sh_coeffs = sh_coeffs.expand(directions.shape[0], -1, -1)
        elif sh_coeffs.shape[0] != directions.shape[0]:
            raise ValueError(
                f"BGField SH coeff batch ({sh_coeffs.shape[0]}) does not match direction batch ({directions.shape[0]})"
            )
        return spherical_harmonics(degrees_to_use=num_sh, dirs=directions, coeffs=sh_coeffs)

    def forward(
        self,
        ray_bundle: RayBundle,
        appearance_embedding: Optional[Tensor] = None,
        num_sh: Optional[int] = None,
    ) -> Tensor:
        if num_sh is None:
            num_sh = int(round(self.sh_dim**0.5)) - 1
        return self.get_background_rgb(
            ray_bundle=ray_bundle,
            appearance_embedding=appearance_embedding,
            num_sh=num_sh,
        )


class SplatfactoWField(Field):
    def __init__(
        self,
        appearance_embed_dim: int,
        appearance_features_dim: int,
        implementation: Literal["tcnn", "torch"] = "torch",
        sh_levels: int = 4,
        num_layers: int = 3,
        layer_width: int = 256,
    ):
        super().__init__()
        self.sh_dim = (sh_levels + 1) ** 2
        self.encoder = MLP(
            in_dim=appearance_embed_dim + appearance_features_dim,
            num_layers=num_layers - 1,
            layer_width=layer_width,
            out_dim=layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation=implementation,
        )
        self.sh_base_head = nn.Linear(layer_width, 3)
        self.sh_rest_head = nn.Linear(layer_width, (self.sh_dim - 1) * 3)
        self.sh_rest_head.weight.data.zero_()
        self.sh_rest_head.bias.data.zero_()

    def _encode(self, appearance_embed: Tensor, appearance_features: Tensor) -> Tensor:
        if appearance_embed.dim() == 1:
            appearance_embed = appearance_embed.unsqueeze(0)
        if appearance_embed.shape[0] == 1 and appearance_features.shape[0] != 1:
            appearance_embed = appearance_embed.expand(appearance_features.shape[0], -1)
        elif appearance_embed.shape[0] != appearance_features.shape[0]:
            raise ValueError(
                f"appearance_embed batch ({appearance_embed.shape[0]}) must be 1 or match appearance_features batch ({appearance_features.shape[0]})"
            )
        x = torch.cat((appearance_embed, appearance_features), dim=-1)
        return self.encoder(x).float()

    def get_sh_coeffs(
        self,
        appearance_embed: Tensor,
        appearance_features: Tensor,
        num_sh: Optional[int] = None,
    ) -> Tensor:
        x = self._encode(appearance_embed, appearance_features)
        full_sh_dim = self.sh_dim
        target_sh_dim = full_sh_dim if num_sh is None else (num_sh + 1) ** 2
        target_sh_dim = min(target_sh_dim, full_sh_dim)

        base_color = self.sh_base_head(x)
        sh_rest = self.sh_rest_head(x)[..., : max(target_sh_dim - 1, 0) * 3]
        return torch.cat([base_color, sh_rest], dim=-1).view(-1, target_sh_dim, 3)

    def forward(
        self,
        appearance_embed: Tensor,
        appearance_features: Tensor,
        num_sh: Optional[int] = None,
    ) -> Tensor:
        return self.get_sh_coeffs(
            appearance_embed=appearance_embed,
            appearance_features=appearance_features,
            num_sh=num_sh,
        )

    def shs_0(self, appearance_embed: Tensor, appearance_features: Tensor) -> Tensor:
        x = self._encode(appearance_embed, appearance_features)
        base_color = self.sh_base_head(x)
        return base_color.view(-1, 3)

    def shs_rest(self, appearance_embed: Tensor, appearance_features: Tensor) -> Tensor:
        x = self._encode(appearance_embed, appearance_features)
        sh_rest = self.sh_rest_head(x)
        return sh_rest.view(-1, self.sh_dim - 1, 3)
