# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors.
# Licensed under the Apache License, Version 2.0.

"""Phototourism / NeRF-W dataset parser for Splatfacto-W.

This version keeps the original training behavior but fixes a few structural issues:
- consistently uses config.colmap_path,
- iterates COLMAP images correctly (image ids != camera ids),
- validates TSV split files more carefully,
- keeps a stable mapping from eval-dataset indices to original image indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import pandas as pd
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class NerfWDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: NerfW)
    data: Path = Path("data/brandenburg-gate")
    data_name: Literal["brandenburg", "trevi", "sacre"] = "brandenburg"
    scale_factor: float = 3.0
    alpha_color: str = "white"
    scene_scale: float = 1.0
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    center_method: Literal["poses", "focus", "none"] = "poses"
    auto_scale_poses: bool = True
    colmap_path: Path = Path("dense/sparse")
    load_3D_points: bool = True
    depth_unit_scale_factor: float = 1e-3
    downscale_factor: Optional[int] = None
    max_2D_matches_per_3D_point: int = 0


@dataclass
class NerfW(DataParser):
    config: NerfWDataParserConfig

    def __init__(self, config: NerfWDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.i_eval: list[int] = []

    def _load_split_file(self) -> pd.DataFrame:
        split_file = self.data / f"{self.config.data_name}.tsv"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing NeRF-W split file: {split_file}")

        split_data = pd.read_csv(split_file, sep="\t").dropna()
        required_cols = {"filename", "split"}
        missing = required_cols - set(split_data.columns)
        if missing:
            raise ValueError(f"Split file {split_file} is missing columns: {sorted(missing)}")
        split_data["filename"] = split_data["filename"].astype(str)
        split_data["split"] = split_data["split"].astype(str).str.lower()
        return split_data

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        image_filenames: list[Path] = []
        poses: list[torch.Tensor] = []
        fxs: list[torch.Tensor] = []
        fys: list[torch.Tensor] = []
        cxs: list[torch.Tensor] = []
        cys: list[torch.Tensor] = []

        colmap_path = self.data / self.config.colmap_path
        cameras_path = colmap_path / "cameras.bin"
        images_path = colmap_path / "images.bin"

        with CONSOLE.status(f"[bold green]Reading phototourism images and poses for {split} split..."):
            cams = colmap_utils.read_cameras_binary(cameras_path)
            imgs = colmap_utils.read_images_binary(images_path)

        split_data = self._load_split_file()
        split_filenames = set(split_data["filename"].tolist())

        # Iterate images, then resolve the owning camera via image.camera_id.
        for _, img in sorted(imgs.items(), key=lambda kv: kv[1].name):
            if img.name not in split_filenames:
                continue
            cam = cams[img.camera_id]
            if cam.model != "PINHOLE":
                raise ValueError(
                    f"Only PINHOLE cameras are currently supported; got {cam.model} for image {img.name}"
                )

            pose = torch.cat(
                [torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))],
                dim=1,
            )
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            poses.append(torch.linalg.inv(pose))
            fxs.append(torch.tensor(cam.params[0]))
            fys.append(torch.tensor(cam.params[1]))
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))
            image_filenames.append(self.data / "dense/images" / img.name)

        if not image_filenames:
            raise ValueError(f"No valid images found for dataset at {self.data}")

        poses_t = torch.stack(poses).float()
        poses_t[..., 1:3] *= -1
        fxs_t = torch.stack(fxs).float()
        fys_t = torch.stack(fys).float()
        cxs_t = torch.stack(cxs).float()
        cys_t = torch.stack(cys).float()

        filename_to_index = {path.name: idx for idx, path in enumerate(image_filenames)}
        eval_names = split_data.loc[split_data["split"] == "test", "filename"].tolist()
        eval_indices = [filename_to_index[name] for name in eval_names if name in filename_to_index]
        eval_indices_t = torch.tensor(eval_indices, dtype=torch.long)
        self.i_eval = eval_indices

        CONSOLE.log(f"eval_indices: {eval_indices_t}")
        CONSOLE.log(f"eval_filenames: {[image_filenames[i] for i in eval_indices]}")

        all_indices = torch.arange(len(image_filenames), dtype=torch.long)
        if split == "train":
            # Preserve original behavior: keep all images in train, while the datamanager
            # applies masking to eval-designated images.
            indices = all_indices
        elif split in {"val", "test"}:
            indices = eval_indices_t
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses_t, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses_t,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses_t[:, :3, 3])).clamp_min(1e-6))
        scale_factor *= self.config.scale_factor
        poses_t[:, :3, 3] *= scale_factor

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]],
                dtype=torch.float32,
            )
        )

        cameras = Cameras(
            camera_to_worlds=poses_t[:, :3, :4],
            fx=fxs_t,
            fy=fys_t,
            cx=cxs_t,
            cy=cys_t,
            camera_type=CameraType.PERSPECTIVE,
        )
        cameras = cameras[indices]
        selected_filenames = [image_filenames[i] for i in indices.tolist()]

        metadata = {}
        if split == "train" and self.config.load_3D_points:
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor))

        if len(cameras) != len(selected_filenames):
            raise RuntimeError("Cameras / image filename count mismatch after split selection")

        return DataparserOutputs(
            image_filenames=selected_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )

    def _load_3D_points(self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        if (colmap_path / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {colmap_path}")

        points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
        points3D = (
            torch.cat((points3D, torch.ones_like(points3D[..., :1])), dim=-1)
            @ transform_matrix.T
        )
        points3D *= scale_factor

        points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))
        points3D_num_points = torch.tensor([len(p.image_ids) for p in colmap_points.values()], dtype=torch.int64)
        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.from_numpy(np.array([p.error for p in colmap_points.values()], dtype=np.float32)),
            "points3D_num_points2D": points3D_num_points,
        }

        if self.config.max_2D_matches_per_3D_point != 0:
            if (colmap_path / "images.txt").exists():
                im_id_to_image = colmap_utils.read_images_text(colmap_path / "images.txt")
            elif (colmap_path / "images.bin").exists():
                im_id_to_image = colmap_utils.read_images_binary(colmap_path / "images.bin")
            else:
                raise ValueError(f"Could not find images.txt or images.bin in {colmap_path}")

            downscale_factor = self._downscale_factor
            max_num_points = int(torch.max(points3D_num_points).item())
            if self.config.max_2D_matches_per_3D_point > 0:
                max_num_points = min(max_num_points, self.config.max_2D_matches_per_3D_point)

            points3D_image_ids = []
            points3D_image_xy = []
            for p in colmap_points.values():
                nids = np.array(p.image_ids, dtype=np.int64)
                nxy_ids = np.array(p.point2D_idxs, dtype=np.int32)
                if self.config.max_2D_matches_per_3D_point != -1:
                    idxs = np.argsort(p.error)[: self.config.max_2D_matches_per_3D_point]
                    nids = nids[idxs]
                    nxy_ids = nxy_ids[idxs]
                nxy = [im_id_to_image[im_id].xys[pt_idx] for im_id, pt_idx in zip(nids, nxy_ids)]
                nxy = torch.from_numpy(np.stack(nxy).astype(np.float32)) if len(nxy) else torch.zeros((0, 2), dtype=torch.float32)
                nids_t = torch.from_numpy(nids)
                points3D_image_ids.append(
                    torch.cat((nids_t, torch.full((max_num_points - len(nids_t),), -1, dtype=torch.int64)))
                )
                points3D_image_xy.append(
                    torch.cat(
                        (
                            nxy,
                            torch.full((max_num_points - len(nxy), 2), 0, dtype=torch.float32),
                        )
                    )
                    / downscale_factor
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)
        return out

    def check_in_eval(self, idx: int) -> bool:
        return idx in self.i_eval

    def find_eval_idx(self, idx: int) -> int:
        return self.i_eval[idx]


splatfactow_dataparser = DataParserSpecification(config=NerfWDataParserConfig())
