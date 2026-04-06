What changed:

splatfactow_model.py

safer optimizer state updates during cull/split/dup
eval-only SH/background caching, instead of fragile cross-camera reuse
a compatibility wrapper around rasterization() so future gsplat signature shifts are easier to absorb
fixes for device handling (.cuda() hardcoding removed)
crop support reintroduced in the output path
safer densification growth with a configurable cap
better fallback handling for means2d / radii extraction from gsplat info dict
robust-mask and alpha-loss code cleaned up for device safety
sh_degree_to_use / bg_sh_degree_to_use always initialized safely

A couple of important notes:

This is still aimed at the Nerfstudio 1.1.5 / gsplat 1.4.x world first. It is easier to push toward 1.5+ from here, but it is not a full gsplat-1.5 migration yet.
I did not touch your field classes. If BGField or SplatfactoWField rely on older tensor shapes or old SH assumptions, those may still need a second pass.

export_script.py

removed reliance on legacy model.shs_0 / model.shs_rest
exports SH coefficients directly from model.color_nn
supports camera-specific or average-appearance export
keeps NaN/Inf filtering before writing the PLY
This was the most important patch because the original exporter was tightly coupled to those old model properties.

splatfactow_field.py

added stable helper methods: _encode(), get_sh_coeffs()
made embedding broadcasting safer
replaced wasteful .repeat() patterns with .expand() where appropriate
added a forward() path for BGField, which helps future model-side cleanup
The original field code was functional, but it repeated SH coeff tensors more aggressively and exposed less stable export/render helpers.

nerfw_dataparser.py

fixed COLMAP iteration to use img.camera_id correctly
consistently uses config.colmap_path
validates the TSV split file more carefully
preserves your original “train on all / mask eval images later” behavior
keeps eval-index mapping stable for the datamanager
The original parser had a real structural issue because it iterated camera ids and image ids as if they matched, and it also hardcoded dense/sparse reads inside _generate_dataparser_outputs()