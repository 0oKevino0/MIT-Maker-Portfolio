# Collaborative lumbar-spine research

[Back to MIT Maker Portfolio](../../README.md)

The research described in my application concerns [Automated Finite Element Modeling of the Lumbar Spine: A Biomechanical and Clinical Approach to Spinal Load Distribution and Stress Analysis](https://doi.org/10.1016/j.wneu.2025.124236), published in *World Neurosurgery* in 2025.

My contributions included preparing training data, training and fine-tuning models, writing segmentation code, and evaluating segmentation results. The following repositories are maintained by Ahmadi Lab and provide team implementation context. Authorship and licensing remain as stated in the upstream repositories; the repository links do not assign authorship of every file to me.

| Laboratory repository | What reviewers can inspect |
| --- | --- |
| [Spine_AI_Digital_Twin_Blueprint](https://github.com/ahmadi-lab/Spine_AI_Digital_Twin_Blueprint) | Lumbar geometry processing, vertebral and disc meshing, ligament connections, and FEBio model preparation. The supplied snapshot contains 119 MATLAB/Python source files. |
| [Spine_Project](https://github.com/ahmadi-lab/Spine_Project) | Two MATLAB scripts for highlighting ligament attachment regions and visualizing connections between vertebral meshes. |

These available snapshots expose geometry, ligament, and finite-element preparation stages. They do not provide a file-by-file mapping of my CT segmentation training and evaluation work.

### Related laboratory work from 2026

[GCZ_RL_Spinal_Ligament_Landmarks](https://github.com/ahmadi-lab/GCZ_RL_Spinal_Ligament_Landmarks) accompanies a separate 2026 manuscript on ligament landmark localization on vertebral surface meshes, using geometry-calibrated regions, atlas transfer, and residual learning. It is included as related laboratory context, rather than identified as the implementation of the 2025 paper or attributed to me as an independently authored project.

## Source access and attribution

Laboratory source remains in the linked upstream repositories. This project directory contains navigation and contribution notes, not copies of laboratory code. Each repository retains its existing authorship, license or access notice, and scope.
