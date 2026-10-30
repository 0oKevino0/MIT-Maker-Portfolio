# Portfolio code guide

## AI CAD Tutor

The [source directory](source/) contains 88 application source and configuration/dependency files for the AI CAD Tutor. I lead its design and development. Claude assisted with portions of the code, frontend work, and review; Codex assisted with version control and review.

Start with the [React interface](source/apps/web/src/App.tsx), [course API](source/services/api/app/routes/course.py), and [teaching-mode logic](source/services/api/app/services/teaching_modes.py). The [README](README.md) describes the working course workflow and the runtime inputs omitted from this public source snapshot. The application video is supplied separately in the portfolio.

## Collaborative lumbar-spine research

The research described in my application concerns [Automated Finite Element Modeling of the Lumbar Spine: A Biomechanical and Clinical Approach to Spinal Load Distribution and Stress Analysis](https://doi.org/10.1016/j.wneu.2025.124236), published in *World Neurosurgery* in 2025.

My contributions included preparing training data, training and fine-tuning models, writing segmentation code, and evaluating segmentation results. The following repositories are maintained by Ahmadi Lab and provide team implementation context. Authorship and licensing remain as stated in the upstream repositories; the repository links do not assign authorship of every file to me.

| Laboratory repository | What reviewers can inspect |
| --- | --- |
| [Spine_AI_Digital_Twin_Blueprint](https://github.com/ahmadi-lab/Spine_AI_Digital_Twin_Blueprint) | Lumbar geometry processing, vertebral and disc meshing, ligament connections, and FEBio model preparation. The supplied snapshot contains 119 MATLAB/Python source files. |
| [Spine_Project](https://github.com/ahmadi-lab/Spine_Project) | Two MATLAB scripts for highlighting ligament attachment regions and visualizing connections between vertebral meshes. |

These available snapshots expose geometry, ligament, and finite-element preparation stages. They do not provide a file-by-file mapping of my CT segmentation training and evaluation work.

### Related laboratory work from 2026

[GCZ_RL_Spinal_Ligament_Landmarks](https://github.com/ahmadi-lab/GCZ_RL_Spinal_Ligament_Landmarks) accompanies a separate 2026 manuscript on ligament landmark localization on vertebral surface meshes, using geometry-calibrated regions, atlas transfer, and residual learning. It is included as related laboratory context, rather than identified as the implementation of the 2025 paper or attributed to me as an independently authored project.

## Adaptive massager design

My [claude-cad](https://github.com/0oKevino0/claude-cad) workflow connects Claude Code with Python/CadQuery/OCP for parametric CAD construction, geometry checks, and export. I used this workflow in the massager design and SOLIDWORKS for assembly. The portfolio's CAD evidence PDF shows the design output; the repository documents the modeling workflow. Cura was used for print preparation.

This tool link describes the design process. It does not establish completed fabrication, feedback-control testing, or clinical effectiveness.

## Source provenance

Laboratory repositories are linked directly, without copying their code into this repository or changing their authorship. Model weights, course materials, private configuration, and laboratory data are not included in this public review snapshot. Each linked repository retains its own terms and scope.
