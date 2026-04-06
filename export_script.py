# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors.
# Licensed under the Apache License, Version 2.0.

"""Export Splatfacto-W checkpoints to PLY.

Retro-compatible patched exporter:
- prefers model.get_sh_coeffs(camera_idx=...) when present,
- falls back to model.color_nn.get_sh_coeffs(...) for forward-looking models,
- falls back again to legacy model.shs_0 / model.shs_rest if needed,
- keeps finite-value filtering before writing the PLY.
"""

from __future__ import annotations

import argparse
import typing
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from splatfactow.splatfactow_model import SplatfactoWModel


@dataclass
class Exporter:
    load_config: Path
    output_dir: Path


@dataclass
class ExportGaussianSplat(Exporter):
    obb_center: Optional[Tuple[float, float, float]] = None
    obb_rotation: Optional[Tuple[float, float, float]] = None
    obb_scale: Optional[Tuple[float, float, float]] = None
    camera_idx: Optional[int] = None

    @staticmethod
    def write_ply(filename: str, count: int, map_to_tensors: typing.OrderedDict[str, np.ndarray]) -> None:
        if not all(len(tensor) == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be non-empty numpy arrays of float or uint8 type")

        with open(filename, "wb") as ply_file:
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            ply_file.write(f"element vertex {count}\n".encode())
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())
            ply_file.write(b"end_header\n")
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    else:
                        ply_file.write(value.tobytes())

    @staticmethod
    def _appearance_embedding_for_export(model: SplatfactoWModel, camera_idx: Optional[int]) -> torch.Tensor:
        if camera_idx is None:
            if getattr(model.config, "use_avg_appearance", False):
                return model.appearance_embeds.weight.mean(dim=0)
            return model.appearance_embeds.weight[0]
        if camera_idx < 0 or camera_idx >= model.num_train_data:
            raise ValueError(f"camera_idx {camera_idx} is out of range [0, {model.num_train_data})")
        return model.appearance_embeds(torch.tensor(camera_idx, device=model.device))

    @staticmethod
    def _normalize_sh_output(sh_coeffs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Normalize SH coeff output to exporter layout."""
        if sh_coeffs.ndim != 3 or sh_coeffs.shape[-1] != 3:
            raise ValueError(f"Unexpected SH coeff shape: {tuple(sh_coeffs.shape)}")
        sh_coeffs = sh_coeffs.contiguous()
        shs_0 = sh_coeffs[:, 0, :].detach().cpu().numpy()  # [N, 3]
        shs_rest = sh_coeffs[:, 1:, :].transpose(1, 2).contiguous().detach().cpu().numpy()  # [N, 3, K-1]
        shs_rest = shs_rest.reshape(shs_rest.shape[0], -1)
        return shs_0, shs_rest

    @classmethod
    def _export_sh_coeffs(cls, model: SplatfactoWModel, appearance_embed: torch.Tensor, camera_idx: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
        # Preferred modern API on the model itself.
        if hasattr(model, "get_sh_coeffs") and callable(getattr(model, "get_sh_coeffs")):
            with torch.no_grad():
                sh_coeffs = model.get_sh_coeffs(cam_idx=camera_idx)
            return cls._normalize_sh_output(sh_coeffs)

        # Forward-looking field-based API.
        color_nn = getattr(model, "color_nn", None)
        appearance_features = getattr(model, "appearance_features", None)
        if color_nn is not None and hasattr(color_nn, "get_sh_coeffs") and appearance_features is not None:
            with torch.no_grad():
                sh_coeffs = color_nn.get_sh_coeffs(
                    appearance_embed=appearance_embed,
                    appearance_features=appearance_features,
                    num_sh=model.config.sh_degree,
                )
            return cls._normalize_sh_output(sh_coeffs)

        # Legacy fallback using model.shs_0 / model.shs_rest properties.
        if hasattr(model, "set_camera_idx") and camera_idx is not None:
            try:
                model.set_camera_idx(camera_idx)
            except Exception:
                pass

        if hasattr(model, "shs_0") and hasattr(model, "shs_rest"):
            with torch.no_grad():
                shs_0_t = model.shs_0
                shs_rest_t = model.shs_rest
            if shs_0_t.ndim == 3 and shs_0_t.shape[1] == 1:
                shs_0 = shs_0_t.squeeze(1).detach().cpu().numpy()
            elif shs_0_t.ndim == 2:
                shs_0 = shs_0_t.detach().cpu().numpy()
            else:
                raise ValueError(f"Unexpected legacy shs_0 shape: {tuple(shs_0_t.shape)}")

            if shs_rest_t.ndim != 3:
                raise ValueError(f"Unexpected legacy shs_rest shape: {tuple(shs_rest_t.shape)}")
            shs_rest = shs_rest_t.transpose(1, 2).contiguous().detach().cpu().numpy()
            shs_rest = shs_rest.reshape(shs_rest.shape[0], -1)
            return shs_0, shs_rest

        raise AttributeError(
            "Could not export SH coefficients: expected one of "
            "model.get_sh_coeffs(...), model.color_nn.get_sh_coeffs(...), "
            "or legacy model.shs_0 / model.shs_rest."
        )

    def main(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _, pipeline, _, _ = eval_setup(self.load_config)
        if not isinstance(pipeline.model, SplatfactoWModel):
            raise TypeError(f"Expected SplatfactoWModel, got {type(pipeline.model).__name__}")

        model: SplatfactoWModel = pipeline.model
        filename = self.output_dir / "splat.ply"
        map_to_tensors: "OrderedDict[str, np.ndarray]" = OrderedDict()

        with torch.no_grad():
            appearance_embed = self._appearance_embedding_for_export(model, self.camera_idx)
            positions = model.means.detach().cpu().numpy()
            count = positions.shape[0]
            n = count

            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if model.config.sh_degree <= 0:
                raise ValueError("SH degree must be greater than 0 for Gaussian export")

            shs_0, shs_rest = self._export_sh_coeffs(model, appearance_embed, self.camera_idx)
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i]
            for i in range(shs_rest.shape[1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i]

            map_to_tensors["opacity"] = model.opacities.detach().cpu().numpy().reshape(-1)
            scales = model.scales.detach().cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i]
            quats = model.quats.detach().cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i]

            if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
                crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                if crop_obb is None:
                    raise ValueError("Failed to construct crop OBB")
                mask = crop_obb.within(torch.from_numpy(positions)).cpu().numpy()
                for key in list(map_to_tensors.keys()):
                    map_to_tensors[key] = map_to_tensors[key][mask]
                count = int(mask.sum())

        select = np.ones(count, dtype=bool)
        for key, tensor in map_to_tensors.items():
            tensor_2d = tensor if tensor.ndim > 1 else tensor[:, None]
            before = int(np.sum(select))
            select = np.logical_and(select, np.isfinite(tensor_2d).all(axis=-1))
            after = int(np.sum(select))
            if after < before:
                CONSOLE.print(f"{before - after} NaN/Inf elements in {key}")

        if int(np.sum(select)) < count:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {int(np.sum(select))}/{count}")
            for key in list(map_to_tensors.keys()):
                map_to_tensors[key] = map_to_tensors[key][select]
            count = int(np.sum(select))

        self.write_ply(str(filename), count, map_to_tensors)
        CONSOLE.print(f"Exported {count} gaussians to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Gaussian Splat model to a .ply")
    parser.add_argument("--load_config", type=Path, help="Path to the config YAML file.")
    parser.add_argument("--output_dir", type=Path, help="Path to the output directory.")
    parser.add_argument("--obb_center", type=str, help="Center of the oriented bounding box.")
    parser.add_argument("--obb_rotation", type=str, help="Rotation of the oriented bounding box in RPY radians.")
    parser.add_argument("--obb_scale", type=str, help="Scale of the oriented bounding box along each axis.")
    parser.add_argument("--camera_idx", type=int, help="Camera index to use for export appearance.")
    args = parser.parse_args()

    obb_center = tuple(map(float, args.obb_center.split(","))) if args.obb_center else None
    obb_rotation = tuple(map(float, args.obb_rotation.split(","))) if args.obb_rotation else None
    obb_scale = tuple(map(float, args.obb_scale.split(","))) if args.obb_scale else None

    exporter = ExportGaussianSplat(
        load_config=args.load_config,
        output_dir=args.output_dir,
        obb_center=obb_center,
        obb_rotation=obb_rotation,
        obb_scale=obb_scale,
        camera_idx=args.camera_idx,
    )
    exporter.main()
