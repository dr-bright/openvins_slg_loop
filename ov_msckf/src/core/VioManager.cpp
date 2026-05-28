/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "track/TrackSLG.h"
#include "track/UpdaterSLGM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {

constexpr int REID_MIN_TRACK_GENERATION = 2;
constexpr double REID_MIN_SAME_POINT_PROBABILITY = 0.80;
constexpr double REID_COVARIANCE_FLOOR_M = 0.50;
constexpr double REID_DIFFERENT_POINT_RADIUS_M = 5.0;

int feature_measurement_count(const std::shared_ptr<Feature> &feat) {
  int count = 0;
  for (const auto &pair : feat->timestamps) {
    count += (int)pair.second.size();
  }
  return count;
}

bool feature_has_current_measurement(const std::shared_ptr<Feature> &feat, const ov_core::CameraData &message) {
  for (const auto &cam_id : message.sensor_ids) {
    if (feat->timestamps.find(cam_id) == feat->timestamps.end()) {
      continue;
    }
    for (const auto &timestamp : feat->timestamps.at(cam_id)) {
      if (std::abs(timestamp - message.timestamp) < 1e-9) {
        return true;
      }
    }
  }
  return false;
}

Eigen::Vector3d landmark_position_in_global(const std::shared_ptr<State> &state, const std::shared_ptr<Landmark> &landmark) {
  Eigen::Vector3d p_FinG = landmark->get_xyz(false);
  if (LandmarkRepresentation::is_relative_representation(landmark->_feat_representation)) {
    assert(landmark->_anchor_cam_id != -1);
    assert(landmark->_anchor_clone_timestamp != -1);
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(landmark->_anchor_cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(landmark->_anchor_cam_id)->pos();
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(landmark->_anchor_clone_timestamp)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(landmark->_anchor_clone_timestamp)->pos();
    p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (landmark->get_xyz(false) - p_IinC) + p_IinG;
  }
  return p_FinG;
}

ov_lightglue::TrackedLandmarkSLGM make_tracked_landmark_slgm(const std::shared_ptr<State> &state,
                                                             const std::shared_ptr<Landmark> &landmark) {
  ov_lightglue::TrackedLandmarkSLGM tracked;
  tracked.landmark = landmark;
  tracked.featid = landmark->_featid;
  tracked.p_FinG = landmark_position_in_global(state, landmark);
  tracked.should_marg = landmark->should_marg;
  tracked.unique_camera_id = landmark->_unique_camera_id;
  tracked.unobserved_count = landmark->unobserved_count;
  tracked.update_fail_count = landmark->update_fail_count;
  return tracked;
}

std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>>
camera_clones_for_state(const std::shared_ptr<State> &state) {
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {
      Eigen::Matrix3d R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Vector3d p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }
    clones_cam.insert({clone_calib.first, clones_cami});
  }
  return clones_cam;
}

bool invert_spd_3x3(const Eigen::Matrix3d &mat, Eigen::Matrix3d &inv) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(mat);
  if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() <= 1e-12) {
    return false;
  }
  inv = solver.eigenvectors() * solver.eigenvalues().cwiseInverse().asDiagonal() * solver.eigenvectors().transpose();
  return true;
}

bool candidate_spatial_hypothesis(const std::shared_ptr<State> &state, const VioManagerOptions &params,
                                  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> &clones_cam,
                                  const std::shared_ptr<Feature> &candidate, Eigen::Vector3d &p_FinG,
                                  Eigen::Matrix3d &cov_FinG, int &anchor_cam_id) {
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  auto feat = std::make_shared<Feature>(*candidate);
  feat->clean_old_measurements(clonetimes);
  if (feature_measurement_count(feat) < REID_MIN_TRACK_GENERATION) {
    return false;
  }

  auto featinit_options = params.featinit_options;
  FeatureInitializer initializer(featinit_options);
  bool success_tri = initializer.config().triangulate_1d ? initializer.single_triangulation_1d(feat, clones_cam)
                                                         : initializer.single_triangulation(feat, clones_cam);
  if (!success_tri) {
    return false;
  }
  bool success_refine = initializer.config().refine_features ? initializer.single_gaussnewton(feat, clones_cam) : true;
  if (!success_refine) {
    return false;
  }

  p_FinG = feat->p_FinG;
  anchor_cam_id = feat->anchor_cam_id;

  UpdaterHelper::UpdaterHelperFeature updater_feat;
  updater_feat.featid = feat->featid;
  updater_feat.uvs = feat->uvs;
  updater_feat.uvs_norm = feat->uvs_norm;
  updater_feat.timestamps = feat->timestamps;
  updater_feat.feat_representation = LandmarkRepresentation::Representation::GLOBAL_3D;
  updater_feat.p_FinG = feat->p_FinG;
  updater_feat.p_FinG_fej = feat->p_FinG;

  Eigen::MatrixXd H_f;
  Eigen::MatrixXd H_x;
  Eigen::VectorXd res;
  std::vector<std::shared_ptr<Type>> Hx_order;
  UpdaterHelper::get_feature_jacobian_full(state, updater_feat, H_f, H_x, res, Hx_order);
  if (H_f.rows() < 3 || H_f.cols() != 3) {
    return false;
  }

  Eigen::Matrix3d info = (1.0 / params.slam_options.sigma_pix_sq) * H_f.transpose() * H_f;
  if (!invert_spd_3x3(info, cov_FinG)) {
    cov_FinG = Eigen::Matrix3d::Identity();
  }
  cov_FinG = 0.5 * (cov_FinG + cov_FinG.transpose());
  return true;
}

bool landmark_covariance_in_global(const std::shared_ptr<State> &state, const std::shared_ptr<Landmark> &landmark,
                                   Eigen::Matrix3d &cov_FinG) {
  UpdaterHelper::UpdaterHelperFeature feat;
  feat.featid = landmark->_featid;
  feat.feat_representation = landmark->_feat_representation;
  if (LandmarkRepresentation::is_relative_representation(landmark->_feat_representation)) {
    feat.anchor_cam_id = landmark->_anchor_cam_id;
    feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
    feat.p_FinA = landmark->get_xyz(false);
    feat.p_FinA_fej = landmark->get_xyz(true);
  } else {
    feat.p_FinG = landmark->get_xyz(false);
    feat.p_FinG_fej = landmark->get_xyz(true);
  }

  Eigen::MatrixXd H_f;
  std::vector<Eigen::MatrixXd> H_x;
  std::vector<std::shared_ptr<Type>> x_order;
  UpdaterHelper::get_feature_jacobian_representation(state, feat, H_f, H_x, x_order);

  std::vector<std::shared_ptr<Type>> cov_order = x_order;
  cov_order.push_back(landmark);
  Eigen::MatrixXd cov = StateHelper::get_marginal_covariance(state, cov_order);

  int jac_cols = 0;
  for (const auto &var : x_order) {
    jac_cols += var->size();
  }
  jac_cols += landmark->size();

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, jac_cols);
  int col = 0;
  for (size_t i = 0; i < x_order.size(); i++) {
    J.block(0, col, 3, x_order.at(i)->size()) = H_x.at(i);
    col += x_order.at(i)->size();
  }
  J.block(0, col, 3, landmark->size()) = H_f;

  cov_FinG = J * cov * J.transpose();
  cov_FinG = 0.5 * (cov_FinG + cov_FinG.transpose());
  return cov_FinG.allFinite();
}

double same_point_probability(const Eigen::Vector3d &p_candidate, const Eigen::Matrix3d &cov_candidate,
                              const Eigen::Vector3d &p_landmark, const Eigen::Matrix3d &cov_landmark) {
  Eigen::Matrix3d S = cov_candidate + cov_landmark +
                      REID_COVARIANCE_FLOOR_M * REID_COVARIANCE_FLOOR_M * Eigen::Matrix3d::Identity();
  S = 0.5 * (S + S.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(S);
  if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() <= 1e-12) {
    return 0.0;
  }

  Eigen::Vector3d residual = p_candidate - p_landmark;
  Eigen::Vector3d inv_weighted = solver.eigenvectors().transpose() * residual;
  double d2 = (inv_weighted.array().square() / solver.eigenvalues().array()).sum();

  const double pi = std::acos(-1.0);
  double det = solver.eigenvalues().prod();
  double gaussian_density = std::exp(-0.5 * d2) / std::sqrt(std::pow(2.0 * pi, 3) * det);
  double uniform_volume = (4.0 / 3.0) * pi * std::pow(REID_DIFFERENT_POINT_RADIUS_M, 3);
  double different_density = 1.0 / uniform_volume;
  return gaussian_density / (gaussian_density + different_density);
}

size_t try_reidentify_inactive_slam_landmarks(const std::shared_ptr<State> &state, const VioManagerOptions &params,
                                              const ov_core::CameraData &message,
                                              std::vector<std::shared_ptr<Feature>> &feats_lost,
                                              std::vector<std::shared_ptr<Feature>> &feats_marg,
                                              std::vector<std::shared_ptr<Feature>> &feats_maxtracks,
                                              const std::shared_ptr<FeatureDatabase> &database) {
  if (state->_options.slam_landmark_policy != StateOptions::SlamLandmarkPolicy::SIMPLE || state->_options.max_slam_features <= 0) {
    return 0;
  }

  struct CandidateInfo {
    std::shared_ptr<Feature> feature;
    Eigen::Vector3d p_FinG;
    Eigen::Matrix3d cov_FinG;
    int anchor_cam_id;
  };
  struct MatchInfo {
    size_t landmark_id;
    size_t candidate_id;
    double probability;
  };

  std::unordered_map<size_t, CandidateInfo> candidates;
  auto clones_cam = camera_clones_for_state(state);
  auto data = database->get_internal_data();
  for (const auto &pair : data) {
    const auto &feature = pair.second;
    if (feature == nullptr || feature->to_delete || state->_features_SLAM.find(feature->featid) != state->_features_SLAM.end()) {
      continue;
    }
    if (feature_measurement_count(feature) < REID_MIN_TRACK_GENERATION || !feature_has_current_measurement(feature, message)) {
      continue;
    }
    Eigen::Vector3d p_FinG;
    Eigen::Matrix3d cov_FinG;
    int anchor_cam_id = -1;
    if (!candidate_spatial_hypothesis(state, params, clones_cam, feature, p_FinG, cov_FinG, anchor_cam_id)) {
      continue;
    }
    candidates.insert({feature->featid, CandidateInfo{feature, p_FinG, cov_FinG, anchor_cam_id}});
  }
  if (candidates.empty()) {
    return 0;
  }

  std::vector<MatchInfo> matches;
  for (const auto &landmark_pair : state->_features_SLAM) {
    const auto &landmark = landmark_pair.second;
    if ((int)landmark->_featid <= 4 * state->_options.max_aruco_features || landmark->unobserved_count <= 0 || landmark->should_marg) {
      continue;
    }
    Eigen::Matrix3d cov_landmark;
    if (!landmark_covariance_in_global(state, landmark, cov_landmark)) {
      continue;
    }
    Eigen::Vector3d p_landmark = landmark_position_in_global(state, landmark);
    for (const auto &candidate_pair : candidates) {
      double probability =
          same_point_probability(candidate_pair.second.p_FinG, candidate_pair.second.cov_FinG, p_landmark, cov_landmark);
      if (probability >= REID_MIN_SAME_POINT_PROBABILITY) {
        matches.push_back({landmark->_featid, candidate_pair.first, probability});
      }
    }
  }

  std::sort(matches.begin(), matches.end(), [](const MatchInfo &a, const MatchInfo &b) { return a.probability > b.probability; });

  std::unordered_set<size_t> used_landmarks;
  std::unordered_set<size_t> used_candidates;
  size_t resumed_count = 0;
  for (const auto &match : matches) {
    if (used_landmarks.find(match.landmark_id) != used_landmarks.end() ||
        used_candidates.find(match.candidate_id) != used_candidates.end() ||
        state->_features_SLAM.find(match.landmark_id) == state->_features_SLAM.end() ||
        state->_features_SLAM.find(match.candidate_id) != state->_features_SLAM.end()) {
      continue;
    }

    auto landmark = state->_features_SLAM.at(match.landmark_id);
    auto candidate = candidates.at(match.candidate_id);
    state->_features_SLAM.erase(match.landmark_id);
    landmark->_featid = match.candidate_id;
    landmark->_unique_camera_id = candidate.anchor_cam_id;
    if (landmark->_unique_camera_id == -1 && !message.sensor_ids.empty()) {
      landmark->_unique_camera_id = (int)message.sensor_ids.at(0);
    }
    landmark->unobserved_count = 0;
    landmark->should_marg = false;
    landmark->update_fail_count = 0;
    state->_features_SLAM.insert({match.candidate_id, landmark});
    used_landmarks.insert(match.landmark_id);
    used_candidates.insert(match.candidate_id);
    resumed_count++;
    PRINT_DEBUG(YELLOW "[SLAM-REID]: rebound inactive landmark %zu to track %zu (p=%.3f)\n" RESET, match.landmark_id,
                match.candidate_id, match.probability);
  }

  if (used_candidates.empty()) {
    return 0;
  }

  auto remove_reidentified = [&used_candidates](std::vector<std::shared_ptr<Feature>> &features) {
    auto it = features.begin();
    while (it != features.end()) {
      if (used_candidates.find((*it)->featid) != used_candidates.end()) {
        it = features.erase(it);
      } else {
        it++;
      }
    }
  };
  remove_reidentified(feats_lost);
  remove_reidentified(feats_marg);
  remove_reidentified(feats_maxtracks);
  return resumed_count;
}

} // namespace

VioManager::VioManager(VioManagerOptions &params_) : thread_init_running(false), thread_init_success(false) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Set the IMU intrinsics
  state->_calib_imu_dw->set_value(params.vec_dw);
  state->_calib_imu_dw->set_fej(params.vec_dw);
  state->_calib_imu_da->set_value(params.vec_da);
  state->_calib_imu_da->set_fej(params.vec_da);
  state->_calib_imu_tg->set_value(params.vec_tg);
  state->_calib_imu_tg->set_fej(params.vec_tg);
  state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
  } else {
    if (params.slg_config.superpoint_onnx_path.empty() || params.slg_config.lightglue_onnx_path.empty()) {
      PRINT_ERROR(RED "VioManager(): SLG selected with use_klt=false, but model paths are empty\n" RESET);
      PRINT_ERROR(RED "please set slg_superpoint_onnx_path and slg_lightglue_onnx_path\n" RESET);
      std::exit(EXIT_FAILURE);
    }
    std::shared_ptr<ov_lightglue::TrackSLG> track_slg = std::make_shared<ov_lightglue::TrackSLG>(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.slg_config);
    trackFEATS = track_slg;
    updaterSLGM = std::make_shared<ov_lightglue::UpdaterSLGM>(track_slg);
  }

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time = state->margtimestep();
  if (oldest_time > state->_timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time + state->_calib_dt_CAMtoIMU->value()(0) - 0.10;
  }
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

void VioManager::process_slgm_landmarks(double timestamp) {
  if (updaterSLGM == nullptr) {
    return;
  }

  std::vector<ov_lightglue::TrackedLandmarkSLGM> tracked_landmarks;
  for (const auto &landmark_pair : state->_features_SLAM) {
    const std::shared_ptr<Landmark> &landmark = landmark_pair.second;
    if (landmark == nullptr || (int)landmark->_featid <= 4 * state->_options.max_aruco_features) {
      continue;
    }
    tracked_landmarks.push_back(make_tracked_landmark_slgm(state, landmark));
  }

  const size_t processed_now = updaterSLGM->process_landmarks(timestamp, std::move(tracked_landmarks));
  if (processed_now > 0) {
    PRINT_DEBUG("[SLGM]: processed=%zu map=%zu descriptors=%zu spawned=%zu merged=%zu\n", processed_now, updaterSLGM->map_size(),
                updaterSLGM->stored_descriptors(), updaterSLGM->spawned_landmarks(), updaterSLGM->merged_landmarks());
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      propagator->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  do_feature_propagate_update(message);
}

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(message);
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    propagator->propagate_and_clone(state, message.timestamp);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks mature enough to be made into SLAM features.
  //
  // Legacy OpenVINS tied this to max_clone_size because only features at the
  // marginalization timestep were considered, and the condition was
  // track_length > max_clone_size. A SLAM feature does not actually need to
  // populate every clone: delayed_init() cleans measurements to existing clone
  // timestamps and builds the linear system only from the clones where the
  // feature was observed. Thus maturity can be decoupled from clone-window size.
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  int slam_min_exposure =
      state->_options.slam_min_feat_exposure > 0 ? state->_options.slam_min_feat_exposure : state->_options.max_clone_size + 1;
  auto feature_meets_slam_exposure = [&](const std::shared_ptr<Feature> &feat) {
    for (const auto &cams : feat->timestamps) {
      if ((int)cams.second.size() >= slam_min_exposure) {
        return true;
      }
    }
    return false;
  };
  auto erase_selected_candidate = [&](std::vector<std::shared_ptr<Feature>> &features, const std::shared_ptr<Feature> &selected) {
    features.erase(std::remove_if(features.begin(), features.end(),
                                  [&](const std::shared_ptr<Feature> &feat) { return feat->featid == selected->featid; }),
                   features.end());
  };
  for (const auto &feat_pair : trackFEATS->get_feature_database()->get_internal_data()) {
    const std::shared_ptr<Feature> &feat = feat_pair.second;
    if (feat == nullptr || feat->to_delete) {
      continue;
    }
    if (state->_features_SLAM.find(feat->featid) != state->_features_SLAM.end()) {
      continue;
    }
    if (feature_meets_slam_exposure(feat)) {
      feats_maxtracks.push_back(feat);
      erase_selected_candidate(feats_lost, feat);
      erase_selected_candidate(feats_marg, feat);
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // If enabled, try to bind short current tracks to inactive in-state landmarks
  // before normal SLAM promotion. This lets resumed tracks reuse existing landmarks.
  size_t resumed_slam = 0;
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay) {
    resumed_slam = try_reidentify_inactive_slam_landmarks(state, params, message, feats_lost, feats_marg, feats_maxtracks,
                                                          trackFEATS->get_feature_database());
  }

  // If extended landmark lifetime is enabled, make room for new SLAM candidates
  // by evicting currently inactive visual landmarks first.
  if (state->_options.slam_landmark_policy != StateOptions::SlamLandmarkPolicy::NONE &&
      state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay && !feats_maxtracks.empty()) {
    int slam_capacity = state->_options.max_slam_features + curr_aruco_tags;
    int available_slots = slam_capacity - (int)state->_features_SLAM.size();
    int desired_slots = (int)feats_maxtracks.size();
    int needed_slots = std::max(0, desired_slots - std::max(0, available_slots));
    if (needed_slots > 0) {
      std::vector<std::shared_ptr<Landmark>> inactive_landmarks;
      for (const auto &landmark : state->_features_SLAM) {
        if ((int)landmark.second->_featid > 4 * state->_options.max_aruco_features && landmark.second->unobserved_count > 0) {
          inactive_landmarks.push_back(landmark.second);
        }
      }
      std::sort(inactive_landmarks.begin(), inactive_landmarks.end(),
                [](const std::shared_ptr<Landmark> &a, const std::shared_ptr<Landmark> &b) {
                  return a->unobserved_count > b->unobserved_count;
                });
      int evict_count = std::min(needed_slots, (int)inactive_landmarks.size());
      for (int i = 0; i < evict_count; i++) {
        inactive_landmarks.at(i)->should_marg = true;
      }
      if (evict_count > 0) {
        process_slgm_landmarks(message.timestamp);
        StateHelper::marginalize_slam(state);
      }
    }
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    bool has_active_feat2 = (feat2 != nullptr && !feat2->to_delete);
    if (has_active_feat2) {
      feats_slam.push_back(feat2);
      landmark.second->unobserved_count = 0;
    }
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (!has_active_feat2 && current_unique_cam) {
      if (state->_options.slam_landmark_policy == StateOptions::SlamLandmarkPolicy::NONE) {
        landmark.second->should_marg = true;
      } else if ((int)landmark.second->_featid > 4 * state->_options.max_aruco_features) {
        landmark.second->unobserved_count++;
      }
    }
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  process_slgm_landmarks(message.timestamp);
  StateHelper::marginalize_slam(state);

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
  updaterMSCKF->update(state, featsup_MSCKF);
  propagator->invalidate_cache();
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    updaterSLAM->update(state, featsup_TEMP);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
    propagator->invalidate_cache();
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT5 = boost::posix_time::microsec_clock::local_time();
  updaterSLAM->delayed_init(state, feats_slam_DELAYED);
  rT6 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  std::unordered_set<size_t> protected_slam_feature_ids;
  for (const auto &landmark : state->_features_SLAM) {
    if ((int)landmark.second->_featid > 4 * state->_options.max_aruco_features) {
      protected_slam_feature_ids.insert(landmark.second->_featid);
    }
  }
  trackFEATS->get_feature_database()->cleanup(protected_slam_feature_ids);
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state);

  // Cleanup any features older than the marginalization time
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep(), protected_slam_feature_ids);
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
  }

  // Finally marginalize the oldest clone if needed
  StateHelper::marginalize_old_clone(state);
  rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }

  // Debug for imu intrinsics
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("q_GYROtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_GYROtoIMU->value()(0), state->_calib_imu_GYROtoIMU->value()(1),
               state->_calib_imu_GYROtoIMU->value()(2), state->_calib_imu_GYROtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("q_ACCtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_ACCtoIMU->value()(0), state->_calib_imu_ACCtoIMU->value()(1),
               state->_calib_imu_ACCtoIMU->value()(2), state->_calib_imu_ACCtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("Dw = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("Dw = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.do_calib_imu_g_sensitivity) {
    PRINT_INFO("Tg = | %.4f,%.4f,%.4f |  %.4f,%.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_tg->value()(0),
               state->_calib_imu_tg->value()(1), state->_calib_imu_tg->value()(2), state->_calib_imu_tg->value()(3),
               state->_calib_imu_tg->value()(4), state->_calib_imu_tg->value()(5), state->_calib_imu_tg->value()(6),
               state->_calib_imu_tg->value()(7), state->_calib_imu_tg->value()(8));
  }
  size_t active_slam = 0;
  size_t inactive_slam = 0;
  size_t aruco_slam = 0;
  for (const auto &landmark : state->_features_SLAM) {
    if ((int)landmark.second->_featid <= 4 * state->_options.max_aruco_features) {
      aruco_slam++;
    } else if (landmark.second->unobserved_count > 0) {
      inactive_slam++;
    } else {
      active_slam++;
    }
  }
  PRINT_INFO(YELLOW "SLAM features in state: active=%zu inactive=%zu aruco=%zu total=%zu resumed=%zu\n" RESET, active_slam,
             inactive_slam, aruco_slam, state->_features_SLAM.size(), resumed_slam);
}
