from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import spherical_harmonics
from pytorch_msssim import SSIM
from torch import Tensor
from torch.nn import Parameter

try:
    from gsplat.rendering import rasterization
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install gsplat>=1.4.0") from exc

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.camera_utils import normalize
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from splatfactow.splatfactow_field import BGField, SplatfactoWField


def quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternions to rotation matrices."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    rot = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return rot.reshape(quats.shape[:-1] + (3, 3))

def random_quat_tensor(num: int, *, device: Optional[torch.device] = None) -> Tensor:
    """Sample random quaternions."""
    u = torch.rand(num, device=device)
    v = torch.rand(num, device=device)
    w = torch.rand(num, device=device)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def resize_image(image: torch.Tensor, d: int) -> torch.Tensor:
    """Area-style downscale, matching Nerfstudio's splatfacto helper."""
    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return (
        F.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d)
        .squeeze(1)
        .permute(1, 2, 0)
    )


@torch_compile()
def get_viewmat(optimized_camera_to_world: Tensor) -> Tensor:
    """Convert c2w to gsplat world2camera matrices."""
    rot = optimized_camera_to_world[:, :3, :3]
    trans = optimized_camera_to_world[:, :3, 3:4]
    rot = rot * torch.tensor([[[1, -1, -1]]], device=rot.device, dtype=rot.dtype)
    rot_inv = rot.transpose(1, 2)
    trans_inv = -torch.bmm(rot_inv, trans)
    viewmat = torch.zeros(rot.shape[0], 4, 4, device=rot.device, dtype=rot.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = rot_inv
    viewmat[:, :3, 3:4] = trans_inv
    return viewmat


@dataclass
class SplatfactoWModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoWModel)
    warmup_length: int = 1000
    refine_every: int = 100
    resolution_schedule: int = 3000
    background_color: Literal["random", "black", "white"] = "random"
    num_downscales: int = 2
    cull_alpha_thresh: float = 0.1
    cull_scale_thresh: float = 0.5
    continue_cull_post_densification: bool = True
    reset_alpha_every: int = 25
    densify_grad_thresh: float = 0.0008
    densify_size_thresh: float = 0.01
    n_split_samples: int = 2
    cull_screen_size: float = 0.15
    split_screen_size: float = 0.05
    stop_screen_size_at: int = 15000
    random_init: bool = False
    num_random: int = 50000
    random_scale: float = 10.0
    ssim_lambda: float = 0.2
    stop_split_at: int = 20000
    use_scale_regularization: bool = False
    max_gauss_ratio: float = 10.0
    output_depth_during_training: bool = False
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    enable_bg_model: bool = True
    bg_num_layers: int = 3
    bg_layer_width: int = 128
    implementation: Literal["tcnn", "torch"] = "torch"
    appearance_embed_dim: int = 48
    app_num_layers: int = 3
    app_layer_width: int = 256
    enable_alpha_loss: bool = True
    appearance_features_dim: int = 72
    enable_robust_mask: bool = True
    robust_mask_percentage: Tuple[float, float] = (0.0, 0.40)
    robust_mask_reset_interval: int = 6000
    never_mask_upper: float = 0.4
    start_robust_mask_at: int = 6000
    sh_degree_interval: int = 2000
    sh_degree: int = 3
    bg_sh_degree: int = 4
    use_avg_appearance: bool = False
    eval_right_half: bool = False
    max_new_gaussians_per_refine: int = 500_000
    """Safety cap for split+duplicate growth during one refine step."""
    use_eval_cache: bool = True
    """Cache per-camera appearance/sh results during eval only."""
    tile_size: int = 16
    """Forward-looking knob kept separate so future gsplat updates are localized."""


class SplatfactoWModel(Model):
    config: SplatfactoWModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)
    def get_sh_coeffs(self, cam_idx: Optional[int] = None) -> torch.Tensor:
        device=self.appearance_features.device
        if cam_idx is not None:
            embed = self.appearance_embeds(torch.tensor(cam_idx, device=self.appearance_features.device))
        elif self.config.use_avg_appearance:
            embed = self.appearance_embeds.weight.mean(dim=0)
        else:
            embed = self.appearance_embeds(torch.tensor(0, device=self.appearance_features.device))

        embed_expanded = embed.unsqueeze(0).expand(self.appearance_features.shape[0], -1)
        return self.color_nn(embed_expanded, self.appearance_features)

    def populate_modules(self) -> None:
        device = torch.device(self.kwargs["device"])
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0].to(device))
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3), device=device) - 0.5) * self.config.random_scale)

        self.xys_grad_norm: Optional[Tensor] = None
        self.vis_counts: Optional[Tensor] = None
        self.max_2Dsize: Optional[Tensor] = None
        self.xys: Optional[Tensor] = None
        self.radii: Optional[Tensor] = None

        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances_t = torch.from_numpy(distances).to(device)
        avg_dist = distances_t.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points, device=device))
        appearance_features = torch.nn.Parameter(torch.zeros((num_points, self.config.appearance_features_dim), device=device))

        if self.seed_points is not None and self.seed_points[1].shape[0] > 0:
            colors = torch.nn.Parameter((self.seed_points[1].to(device).float() / 255.0).clamp(0.0, 1.0))
        else:
            colors = torch.nn.Parameter(torch.zeros((num_points, 3), device=device))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1, device=device)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "appearance_features": appearance_features,
                "colors": colors,
                "opacities": opacities,
            }
        )

        self.camera_optimizer = self.config.camera_optimizer.setup(num_cameras=self.num_train_data, device="cpu")

        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.max_loss = 0.0
        self.min_loss = 1e10
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self.optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor([0.1490, 0.1647, 0.2157], device=device)
        else:
            self.background_color = get_color(self.config.background_color).to(device)

        self.appearance_embeds = torch.nn.Embedding(self.num_train_data, self.config.appearance_embed_dim).to(device)
        self.bg_model = (
            BGField(
                appearance_embedding_dim=self.config.appearance_embed_dim,
                implementation=self.config.implementation,
                sh_levels=self.config.bg_sh_degree,
                num_layers=self.config.bg_num_layers,
                layer_width=self.config.bg_layer_width,
            ).to(device)
            if self.config.enable_bg_model
            else None
        )
        self.color_nn = SplatfactoWField(
            appearance_embed_dim=self.config.appearance_embed_dim,
            appearance_features_dim=self.config.appearance_features_dim,
            implementation=self.config.implementation,
            sh_levels=self.config.sh_degree,
            num_layers=self.config.app_num_layers,
            layer_width=self.config.app_layer_width,
        ).to(device)

        self._eval_cache: Dict[str, Any] = {"cam_idx": None, "num_points": None, "colors": None, "bg_sh": None}
        self.camera_idx = 0
        self.last_size = (1, 1)

    def set_camera_idx(self, cam_idx: int) -> None:
        self.camera_idx = int(cam_idx)

    @property
    def num_points(self) -> int:
        return self.means.shape[0]

    @property
    def means(self) -> Parameter:
        return self.gauss_params["means"]

    @property
    def scales(self) -> Parameter:
        return self.gauss_params["scales"]

    @property
    def quats(self) -> Parameter:
        return self.gauss_params["quats"]

    @property
    def appearance_features(self) -> Parameter:
        return self.gauss_params["appearance_features"]

    @property
    def base_colors(self) -> Parameter:
        return self.gauss_params["colors"]

    @property
    def opacities(self) -> Parameter:
        return self.gauss_params["opacities"]

    def load_state_dict(self, state_dict, **kwargs):  # type: ignore[override]
        self.step = 30000
        if "means" in state_dict:
            for name in ["means", "scales", "quats", "appearance_features", "opacities", "colors"]:
                state_dict[f"gauss_params.{name}"] = state_dict[name]
        new_count = state_dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            new_shape = (new_count,) + param.shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device, dtype=param.dtype))
        super().load_state_dict(state_dict, **kwargs)
        self._reset_eval_cache()

    def k_nearest_sklearn(self, x: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        x_np = x.detach().cpu().numpy()
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)
        distances, indices = nn_model.kneighbors(x_np)
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def _reset_eval_cache(self) -> None:
        self._eval_cache = {"cam_idx": None, "num_points": None, "colors": None, "bg_sh": None}

    def set_crop(self, crop_box: Optional[OrientedBox]) -> None:
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor) -> None:
        assert background_color.shape == (3,)
        self.background_color = background_color.to(self.device)

    @staticmethod
    def _metadata_cam_idx(camera: Cameras) -> Optional[int]:
        if camera.metadata is None or "cam_idx" not in camera.metadata:
            return None
        cam_idx = camera.metadata["cam_idx"]
        if isinstance(cam_idx, torch.Tensor):
            return int(cam_idx.item())
        return int(cam_idx)

    def _appearance_embed_for_camera(self, camera: Cameras) -> Tuple[torch.Tensor, Optional[int]]:
        cam_idx = self._metadata_cam_idx(camera)
        if cam_idx is not None:
            return self.appearance_embeds(torch.tensor(cam_idx, device=self.device)), cam_idx
        if self.config.use_avg_appearance:
            return self.appearance_embeds.weight.mean(dim=0), None
        return self.appearance_embeds(torch.tensor(0, device=self.device)), None

    def _should_use_eval_cache(self, cam_idx: Optional[int]) -> bool:
        if self.training or not self.config.use_eval_cache or cam_idx is None:
            return False
        return (
            self._eval_cache["cam_idx"] == cam_idx
            and self._eval_cache["num_points"] == self.num_points
        )

    def _compute_colors(self, appearance_embed: torch.Tensor, use_cache: bool) -> torch.Tensor:
        if use_cache and self._eval_cache["colors"] is not None:
            return self._eval_cache["colors"]
        embed = appearance_embed.unsqueeze(0).expand(self.appearance_features.shape[0], -1)
        colors = self.color_nn(embed, self.appearance_features).float()
        if not self.training:
            self._eval_cache["colors"] = colors
            self._eval_cache["num_points"] = self.num_points
        return colors

    def _compute_bg_sh(self, appearance_embed: torch.Tensor, use_cache: bool) -> Optional[torch.Tensor]:
        if self.bg_model is None:
            return None
        if use_cache and self._eval_cache["bg_sh"] is not None:
            return self._eval_cache["bg_sh"]
        bg_sh = self.bg_model.get_sh_coeffs(appearance_embedding=appearance_embed)
        if not self.training:
            self._eval_cache["bg_sh"] = bg_sh
            self._eval_cache["num_points"] = self.num_points
        return bg_sh

    def _resolve_means2d_and_radii(self, info: Dict[str, Any]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        means2d = info.get("means2d")
        if means2d is None:
            means2d = info.get("gaussian_ids")  # forward-looking fallback; may remain None
        radii = info.get("radii")
        if isinstance(radii, torch.Tensor) and radii.dim() > 1:
            radii = radii[0]
        return means2d, radii

    def _rasterize(
        self,
        *,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        viewmats: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
        render_mode: str,
        sh_degree: int,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        kwargs = dict(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
        )
        try:
            return rasterization(tile_size=self.config.tile_size, **kwargs)
        except TypeError:
            # Forward-looking fallback if tile_size disappears in a future gsplat.
            return rasterization(**kwargs)

    def _replace_optimizer_param(self, optimizer: torch.optim.Optimizer, new_param: Parameter, old_mask: Optional[torch.Tensor] = None) -> None:
        old_param = optimizer.param_groups[0]["params"][0]
        state = optimizer.state.pop(old_param, {})
        new_state: Dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                if old_mask is not None and value.shape[:1] == old_mask.shape[:1]:
                    new_state[key] = value[old_mask]
                else:
                    new_state[key] = value
            else:
                new_state[key] = value
        optimizer.param_groups[0]["params"][0] = new_param
        optimizer.state[new_param] = new_state

    def remove_from_optim(self, optimizer: torch.optim.Optimizer, keep_mask: torch.Tensor, new_param: Parameter) -> None:
        self._replace_optimizer_param(optimizer, new_param, old_mask=keep_mask)

    def remove_from_all_optim(self, optimizers: Optimizers, keep_mask: torch.Tensor) -> None:
        for group, params in self.get_gaussian_param_groups().items():
            self.remove_from_optim(optimizers.optimizers[group], keep_mask, params[0])
        torch.cuda.empty_cache()

    def _append_optimizer_zeros(self, optimizer: torch.optim.Optimizer, old_count: int, new_param: Parameter) -> None:
        old_param = optimizer.param_groups[0]["params"][0]
        state = optimizer.state.pop(old_param, {})
        new_state: Dict[str, Any] = {}
        new_count = new_param.shape[0]
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape[:1] == (old_count,):
                pad_shape = (new_count - old_count,) + value.shape[1:]
                pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
                new_state[key] = torch.cat([value, pad], dim=0)
            else:
                new_state[key] = value
        optimizer.param_groups[0]["params"][0] = new_param
        optimizer.state[new_param] = new_state

    def sync_optimizers_after_append(self, optimizers: Optimizers, old_count: int) -> None:
        for group, params in self.get_gaussian_param_groups().items():
            self._append_optimizer_zeros(optimizers.optimizers[group], old_count, params[0])

    @torch.no_grad()
    def after_train(self, step: int) -> None:
        assert step == self.step
        if self.step >= self.config.stop_split_at or self.xys is None or self.radii is None:
            return
        visible_mask = (self.radii > 0).flatten()
        if not torch.any(visible_mask):
            return
        grad_source = getattr(self.xys, "absgrad", None)
        if grad_source is None:
            grad_source = self.xys.grad
        if grad_source is None:
            return
        grads = grad_source[0][visible_mask].norm(dim=-1).clamp_max(1e3)
        if self.xys_grad_norm is None:
            self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
            self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)
        assert self.vis_counts is not None
        self.vis_counts[visible_mask] += 1
        self.xys_grad_norm[visible_mask] += grads
        if self.max_2Dsize is None:
            self.max_2Dsize = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
        new_radii = self.radii.detach()[visible_mask]
        self.max_2Dsize[visible_mask] = torch.maximum(
            self.max_2Dsize[visible_mask],
            new_radii / float(max(self.last_size[0], self.last_size[1])),
        )

    @torch.no_grad()
    def refinement_after(self, optimizers: Optimizers, step: int) -> None:
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        reset_interval = self.config.reset_alpha_every * self.config.refine_every
        do_densification = (
            self.step < self.config.stop_split_at
            and self.step % reset_interval > self.num_train_data + self.config.refine_every
        )
        if do_densification:
            assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
            avg_grad_norm = ((self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])).clamp_max(10.0)
            high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
            splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze() & high_grads
            if self.step < self.config.stop_screen_size_at:
                splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze() & high_grads
            dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze() & high_grads

            split_count = int(splits.sum().item())
            dup_count = int(dups.sum().item())
            proposed_new = split_count * self.config.n_split_samples + dup_count
            if proposed_new > self.config.max_new_gaussians_per_refine:
                keep_fraction = self.config.max_new_gaussians_per_refine / max(proposed_new, 1)
                split_budget = max(1, int(split_count * keep_fraction)) if split_count > 0 else 0
                dup_budget = max(0, self.config.max_new_gaussians_per_refine - split_budget * self.config.n_split_samples)
                if split_count > split_budget:
                    idx = torch.where(splits)[0][:split_budget]
                    limited = torch.zeros_like(splits)
                    limited[idx] = True
                    splits = limited
                if dup_count > dup_budget:
                    idx = torch.where(dups)[0][:dup_budget]
                    limited = torch.zeros_like(dups)
                    limited[idx] = True
                    dups = limited

            split_params = self.split_gaussians(splits, self.config.n_split_samples)
            dup_params = self.dup_gaussians(dups)
            old_count = self.num_points
            for name, param in self.gauss_params.items():
                self.gauss_params[name] = torch.nn.Parameter(torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0))
            if self.max_2Dsize is not None:
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros(split_params["means"].shape[0], device=self.device),
                        torch.zeros(dup_params["means"].shape[0], device=self.device),
                    ],
                    dim=0,
                )
            self.sync_optimizers_after_append(optimizers, old_count)
            keep_mask = ~torch.cat(
                [
                    splits,
                    torch.zeros(split_params["means"].shape[0] + dup_params["means"].shape[0], device=self.device, dtype=torch.bool),
                ],
                dim=0,
            )
            keep_mask = self.cull_gaussians(keep_mask=keep_mask)
        elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
            keep_mask = self.cull_gaussians()
        else:
            keep_mask = None

        if keep_mask is not None:
            self.remove_from_all_optim(optimizers, keep_mask)

        if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
            reset_value = self.config.cull_alpha_thresh * 2.0
            self.opacities.data = torch.clamp(
                self.opacities.data,
                max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
            )
            optim = optimizers.optimizers["opacities"]
            param = optim.param_groups[0]["params"][0]
            state = optim.state.get(param, None)
            if state is not None and "exp_avg" in state:
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()

        self.xys_grad_norm = None
        self.vis_counts = None
        self.max_2Dsize = None
        self._reset_eval_cache()

    @torch.no_grad()
    def cull_gaussians(self, keep_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        n_before = self.num_points
        keep = (torch.sigmoid(self.opacities) >= self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = int((~keep).sum().item())
        too_big_count = 0
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            too_big = (torch.exp(self.scales).max(dim=-1).values <= self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at and self.max_2Dsize is not None:
                too_big &= (self.max_2Dsize <= self.config.cull_screen_size).squeeze()
            too_big_count = int((~too_big).sum().item())
            keep &= too_big
        if keep_mask is not None:
            keep &= keep_mask
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[keep])
        CONSOLE.log(
            f"Culled {n_before - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {too_big_count} too big, {self.num_points} remaining)"
        )
        self._reset_eval_cache()
        return keep

    @torch.no_grad()
    def split_gaussians(self, split_mask: torch.Tensor, samples: int) -> Dict[str, Tensor]:
        n_splits = int(split_mask.sum().item())
        if n_splits == 0:
            return {name: param[:0].clone() for name, param in self.gauss_params.items()}
        CONSOLE.log(f"Splitting {n_splits}/{self.num_points} gaussians")
        centered = torch.randn((samples * n_splits, 3), device=self.device)
        scaled = torch.exp(self.scales[split_mask]).repeat(samples, 1) * centered
        quats = F.normalize(self.quats[split_mask], dim=-1)
        rots = quat_to_rotmat(quats.repeat(samples, 1))
        rotated = torch.bmm(rots, scaled[..., None]).squeeze(-1)
        new_means = rotated + self.means[split_mask].repeat(samples, 1)
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samples, 1)
        self.scales.data[split_mask] = torch.log(torch.exp(self.scales.data[split_mask]) / size_fac)
        out = {
            "means": new_means,
            "appearance_features": self.appearance_features[split_mask].repeat(samples, 1),
            "colors": self.base_colors[split_mask].repeat(samples, 1),
            "opacities": self.opacities[split_mask].repeat(samples, 1),
            "scales": new_scales,
            "quats": self.quats[split_mask].repeat(samples, 1),
        }
        return out

    @torch.no_grad()
    def dup_gaussians(self, dup_mask: torch.Tensor) -> Dict[str, Tensor]:
        n_dups = int(dup_mask.sum().item())
        if n_dups == 0:
            return {name: param[:0].clone() for name, param in self.gauss_params.items()}
        CONSOLE.log(f"Duplicating {n_dups}/{self.num_points} gaussians")
        return {name: param[dup_mask].clone() for name, param in self.gauss_params.items()}

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        return [
            TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb),
            TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION], self.after_train),
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            ),
        ]

    def step_cb(self, step: int) -> None:
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "appearance_features", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=groups)
        if self.config.enable_bg_model and self.bg_model is not None:
            groups["field_background_encoder"] = list(self.bg_model.encoder.parameters())
            groups["field_background_base"] = list(self.bg_model.sh_base_head.parameters())
            groups["field_background_rest"] = list(self.bg_model.sh_rest_head.parameters())
        groups["appearance_embed"] = list(self.appearance_embeds.parameters())
        groups["appearance_model_encoder"] = list(self.color_nn.encoder.parameters())
        groups["appearance_model_base"] = list(self.color_nn.sh_base_head.parameters())
        groups["appearance_model_rest"] = list(self.color_nn.sh_rest_head.parameters())
        return groups

    def _get_downscale_factor(self) -> int:
        if self.training:
            return 2 ** max(self.config.num_downscales - self.step // self.config.resolution_schedule, 0)
        return 1

    def _downscale_if_required(self, image: torch.Tensor) -> torch.Tensor:
        d = self._get_downscale_factor()
        return resize_image(image, d) if d > 1 else image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self) -> Tensor:
        if self.config.background_color == "random":
            return torch.rand(3, device=self.device) if self.training else self.background_color.to(self.device)
        if self.config.background_color == "white":
            return torch.ones(3, device=self.device)
        if self.config.background_color == "black":
            return torch.zeros(3, device=self.device)
        raise ValueError(f"Unknown background color {self.config.background_color}")

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        if not isinstance(camera, Cameras):
            CONSOLE.log("Called get_outputs with not a camera")
            return {}

        sh_degree_to_use = 0
        bg_sh_degree_to_use = 0
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            bg_interval = max(self.config.sh_degree_interval // 2, 1)
            bg_sh_degree_to_use = min(self.step // bg_interval, self.config.bg_sh_degree)

        appearance_embed, cam_idx = self._appearance_embed_for_camera(camera)
        use_cache = self._should_use_eval_cache(cam_idx)
        if cam_idx is not None and not self.training:
            self._eval_cache["cam_idx"] = cam_idx

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        crop_ids: Optional[torch.Tensor]
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), self.background_color)
        else:
            crop_ids = None

        if crop_ids is None:
            means = self.means
            opacities = self.opacities
            scales = self.scales
            quats = self.quats
            appearance_features = self.appearance_features
            colors_param = self.base_colors
        else:
            means = self.means[crop_ids]
            opacities = self.opacities[crop_ids]
            scales = self.scales[crop_ids]
            quats = self.quats[crop_ids]
            appearance_features = self.appearance_features[crop_ids]
            colors_param = self.base_colors[crop_ids]

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        intrinsics = camera.get_intrinsics_matrices().to(self.device)
        width, height = int(camera.width.item()), int(camera.height.item())
        self.last_size = (height, width)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        colors = self._compute_colors(appearance_embed, use_cache)
        if crop_ids is not None:
            colors = colors[crop_ids]
        if colors.shape[0] != appearance_features.shape[0]:
            # safety fallback for stale eval cache
            embed = appearance_embed.unsqueeze(0).expand(appearance_features.shape[0], -1)
            colors = self.color_nn(embed, appearance_features).float()
            if not self.training:
                self._eval_cache["colors"] = None

        render, alpha, info = self._rasterize(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmat,
            intrinsics=intrinsics,
            width=width,
            height=height,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
        )
        means2d, radii = self._resolve_means2d_and_radii(info)
        if self.training and isinstance(means2d, torch.Tensor) and means2d.requires_grad:
            means2d.retain_grad()
        self.xys = means2d
        self.radii = radii
        alpha = alpha[:, ...]

        if self.bg_model is not None:
            directions = normalize(camera.generate_rays(camera_indices=0, keep_shape=False).directions)
            bg_sh = self._compute_bg_sh(appearance_embed, use_cache)
            assert bg_sh is not None
            background = spherical_harmonics(
                degrees_to_use=bg_sh_degree_to_use,
                coeffs=bg_sh.repeat(directions.shape[0], 1, 1),
                dirs=directions,
            ).view(1, height, width, 3)
        else:
            background = self._get_background_color().view(1, 1, 1, 3)

        rgb = torch.clamp(render[:, ..., :3] + (1 - alpha) * background, 0.0, 1.0)
        camera.rescale_output_resolution(camera_scale_fac)

        if render_mode == "RGB+ED":
            depth = render[:, ..., 3:4]
            depth = torch.where(alpha > 0, depth, depth.detach().max()).squeeze(0)
        else:
            depth = None

        bg_out = background.squeeze(0)
        return {
            "rgb": rgb.squeeze(0),
            "depth": depth,
            "accumulation": alpha.squeeze(0),
            "background": bg_out,
            "base_colors": colors_param,
        }

    def get_gt_img(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return self._downscale_if_required(image).to(self.device)

    def composite_with_background(self, image: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        if image.shape[2] == 4:
            alpha = image[..., -1:].repeat(1, 1, 3)
            return alpha * image[..., :3] + (1 - alpha) * background
        return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        metrics = {
            "psnr": self.psnr(predicted_rgb, gt_rgb),
            "gaussian_count": torch.tensor(float(self.num_points), device=self.device),
        }
        self.camera_optimizer.get_metrics_dict(metrics)
        return metrics

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        l1_img = torch.abs(gt_img - pred_img)
        if self.step >= self.config.start_robust_mask_at and self.config.enable_robust_mask:
            robust_mask = self.robust_mask(l1_img)
            gt_img = gt_img * robust_mask
            pred_img = pred_img * robust_mask
            l1 = (l1_img * robust_mask).mean()
        else:
            l1 = l1_img.mean()

        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio, device=self.device),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0, device=self.device)

        if self.config.enable_alpha_loss:
            alpha_loss = torch.tensor(0.0, device=self.device)
            background = outputs["background"]
            alpha = outputs["accumulation"]
            bg_mask = torch.abs(gt_img - background).mean(dim=-1, keepdim=True) < 0.003
            filt = 3
            window = (torch.ones((filt, filt), device=self.device).view(1, 1, filt, filt) / (filt * filt))
            bg_mask = (
                F.conv2d(bg_mask.float().unsqueeze(0).permute(0, 3, 1, 2), window, stride=1, padding="same")
                .permute(0, 2, 3, 1)
                .squeeze(0)
            )
            alpha_mask = bg_mask > 0.6
            if torch.any(alpha_mask):
                alpha_loss = alpha[alpha_mask].mean() * 0.15
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * l1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
            "alpha_loss": alpha_loss,
        }
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        assert camera is not None
        self.set_crop(obb_box)
        return self.get_outputs(camera.to(self.device))  # type: ignore[return-value]

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        if self.config.eval_right_half:
            gt_rgb = gt_rgb[:, gt_rgb.shape[1] // 2 :, :]
            predicted_rgb = predicted_rgb[:, predicted_rgb.shape[1] // 2 :, :]

        gt_metric = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        pred_metric = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
        psnr = self.psnr(gt_metric, pred_metric)
        ssim = self.ssim(gt_metric, pred_metric)
        lpips = self.lpips(gt_metric, pred_metric)
        metrics = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
        return metrics, {"img": combined_rgb}

    @torch.no_grad()
    def robust_mask(self, errors: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-3
        errors = errors.clone()
        errors[: int(errors.shape[0] * self.config.never_mask_upper), :, :] = 0.0
        l1 = errors.mean()
        if l1 > self.max_loss or self.step % self.config.robust_mask_reset_interval == 0:
            self.max_loss = float(l1)
        if l1 < self.min_loss:
            self.min_loss = float(l1)

        mask_min, mask_max = self.config.robust_mask_percentage
        mask_percentage = (float(l1) - self.min_loss) / ((self.max_loss - self.min_loss) + 1e-6)
        mask_percentage = mask_percentage * (mask_max - mask_min) + mask_min

        errors_b = errors.view(1, errors.shape[0], errors.shape[1], errors.shape[2])
        error_per_pixel = torch.mean(errors_b, dim=-1, keepdim=True)
        inlier_threshold = torch.quantile(error_per_pixel, 1 - mask_percentage)
        is_inlier = (error_per_pixel <= inlier_threshold).float()
        filt = 5
        window = (torch.ones((filt, filt), device=self.device).view(1, 1, filt, filt) / (filt * filt))
        has_inlier_neighbors = F.conv2d(is_inlier.permute(0, 3, 1, 2), window, stride=1, padding="same")
        has_inlier_neighbors = has_inlier_neighbors.permute(0, 2, 3, 1)
        has_inlier_neighbors = (has_inlier_neighbors > 0.4).float()
        return ((has_inlier_neighbors + is_inlier > epsilon).float().view(errors.shape[0], errors.shape[1], 1))

    @torch.no_grad()
    def render_equirect(self, width: int, appearance_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        height = width // 2
        fx = fy = torch.tensor(height, device=self.device)
        cx = torch.tensor(width / 2, device=self.device)
        cy = torch.tensor(height / 2, device=self.device)
        from nerfstudio.cameras.cameras import CameraType

        c2w = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            device=self.device,
            dtype=torch.float32,
        )[None, :3, :]
        camera = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_worlds=c2w,
            camera_type=CameraType.EQUIRECTANGULAR,
        ).to(self.device)
        ray_bundle = camera.generate_rays(0, keep_shape=False, disable_distortion=True)
        assert self.bg_model is not None
        if appearance_embed is None:
            appearance_embed = self.appearance_embeds.weight.mean(dim=0)
        return self.bg_model(ray_bundle, appearance_embed).float().clamp(0, 1).reshape(height, width, 3)
