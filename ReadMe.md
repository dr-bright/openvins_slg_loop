# openvins_slg_loop
This repo is a fork of OpenVINS aimed at replacing standard KLT visual front end with SLG tracker based on SuperPoint + LightGlue combination of AI-models and studying loop closure possibilities.
It aims to improve robostness of vio against bad conditions / bad data. On MH_05_difficult track unfiltered SLG demonstrates significant increase in precision.
Other experimental effors are also tried in this repo, like Grounding DINO semantic feature tracking and experimental SLG-based manual loop closure.


## Project status

The project was actively developed in spring 2026, as of 09.06.2026 all the work is paused at least until late fall 2026. The author says hi to everyone interested in the subject and is open for discussion anytime :>
The project developent during that phase utilized GPT5.5-driven codex agent heavily.

## Core features

-# TrackSLG, a fully-working and tested AI feature tracker that performs no worse than KLT, even better in bad visibility and high blur scenarios.
-# TrackDINO, semantic feature tracker based on grounding dino model. Is untested as a primary tracker, but works standalone.
-# Expanded tooling: ov_eval animate.py, better pose_to_file, pointcloud_to_file.
-# ROS1Visualizer publishes rich pointcloud into /ov_msckf/points_slam, with u,v and slg_desc_xxx, allowing for postprocessing.
-# manual_loop_closure gui app allows for slg-based loop closure in post-processing. Estimated trajectory is corrected using manual frames selection. slg_backend is used to match the two frames to each other. The app takes data from traj.txt and points.pcd dumped from rich points_slam pointcloud with pointcloud_to_file.

## Building

No comprehensive instructions are provided yet, please use the provided dockerfile and docker compose as references.

## Licensing
The codebase and documentation is licensed under the [GNU General Public License v3 (GPL-3)](https://www.gnu.org/licenses/gpl-3.0.txt).
You must preserve the copyright and license notices in your derivative work and make available the complete source code with modifications under the same license ([see this](https://choosealicense.com/licenses/gpl-3.0/); this is not legal advice).


