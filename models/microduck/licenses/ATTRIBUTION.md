# Attribution and license provenance

- Robot design and original 3D files: **Pollen Robotics**, <https://github.com/pollen-robotics/microduck> and <https://github.com/pollen-robotics/microduck_rl>.
- Course canonical MJCF and its two simplified sole collision meshes: **Motphys Technology Co., Ltd.**, MotrixLab commit `1ddb9743c220c3a1e234ea0237d5ce9cc56d0499`, <https://github.com/Motphys/MotrixLab>.
- Original model license text copied from Pollen MicroDuck RL commit `bae39f5900b318f7e87fc485c30a31213cfa14eb`, `LICENSE-MODELS.md`: <https://github.com/pollen-robotics/microduck_rl/blob/bae39f5900b318f7e87fc485c30a31213cfa14eb/LICENSE-MODELS.md>.

Code and MJCF XML: Apache License 2.0; see `LICENSE-APACHE-2.0.txt`.

All `mjcf/assets/*.stl`, including MotrixLab's decimated collision derivatives: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. See `POLLEN-LICENSE-MODELS.md` and <https://creativecommons.org/licenses/by-nc-sa/4.0/>; legal code: <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>.

This course package has **not edited the 40 mesh files or the native XML**. It downloaded the real binary mesh data matching the source Git LFS object hashes, generated `urdf/microduck.urdf` and its metadata from that XML, and added conversion and validation tools. URDF references the original meshes; conversion of a description does not remove the meshes' NonCommercial/ShareAlike conditions. MotrixLab's upstream README records that its sole collision meshes are decimated convex derivatives and that the visual geoms are unmodified.

The generated description and helper scripts are supplied under Apache-2.0 for this teacher package; this statement does not relicense any upstream file or mesh asset. Attribution, licensing and authorization must accompany any redistributed image containing these assets.
