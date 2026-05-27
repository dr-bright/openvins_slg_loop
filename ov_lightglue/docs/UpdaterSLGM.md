# UpdaterSLGM Design Notes

This document records the current working design for `UpdaterSLGM`, a future
persistent map layer for SLG-based OpenVINS. It is intentionally a design note,
not a finished API contract. The goal is to preserve the reasoning, false
starts, and invariants we care about.

## Motivation

OpenVINS has MSCKF features and in-state SLAM landmarks, but it does not maintain
a persistent out-of-view visual map like a keyframe SLAM system. Normal optical
SLAM landmarks are EKF state variables with finite lifetime. When a feature track
dies or the EKF landmark is marginalized, that visual identity normally
disappears.

This is painful on aggressive GoPro drone footage such as `gopro10/fast_hit`.
The estimator can work before an impact, but after a hit it may lose the
coordinate base, reject visual measurements, and stop registering stable
landmarks. Better frame-to-frame tracking helped only at the margin. We need a
persistent visual memory.

`UpdaterSLGM` is the proposed bridge between short-lived EKF SLAM landmarks and
a persistent SLG map.

## Layering

The persistent map must be separate from the EKF state.

```text
SLG feature tracks
  2D observations and SuperPoint descriptors

EKF SLAM landmarks
  in-state 3D landmarks owned by OpenVINS, finite lifetime

SLGM map landmarks
  out-of-state persistent 3D landmarks with descriptor dictionaries
```

The map is populated from EKF SLAM landmarks, not directly from raw tracks:

```text
raw SLG tracks -> EKF SLAM landmarks -> persistent SLGM map
```

This keeps OpenVINS' geometric checks as the quality gate. Raw tracks can be
cheap and disposable; EKF landmarks are the stable points we want to remember.

## Scope

V1 is SLG-only:

- no DINO
- no object classes
- no bounding boxes
- no semantic descriptors

Only SuperPoint descriptors, SLG tracks, EKF landmarks, and persistent point-map
landmarks are considered.

The broader track-capture mechanism should not be SLG-only. It should also work
for future `UpdaterKLTM` and possible ORB-style map updaters. KLT has no
descriptors, but it still benefits from preserving the full observation history
of the track that produced an EKF landmark.

## Spawn Timing

The chosen v1 policy is:

```text
Spawn or update a persistent map landmark when an EKF SLAM landmark is about to
be marginalized.
```

This avoids the complexity of creating map landmarks at EKF-landmark birth and
then managing "growing" vs "mature" states.

Rejected growing-landmark design:

- create map landmark when EKF landmark is born
- append observations while it lives
- mark it mature when the EKF landmark dies
- merge it away if the EKF landmark later matches an existing map landmark

That solves descriptor lifetime naturally, but it adds too many states and edge
cases. For v1, the map has no growing/mature distinction.

## Track Capture Problem

If map landmarks are spawned at EKF landmark death, the feature track that
created the EKF landmark must still be available at death time.

Native OpenVINS aggressively cleans `FeatureDatabase`:

- `Feature::to_delete` means the current active measurements have been consumed.
- `FeatureDatabase::cleanup()` erases features with `to_delete == true`.
- `cleanup_measurements()` erases old measurements outside the clone window.

For persistent mapping we need a broader rule:

```text
Do not erase a feature object while an EKF landmark still depends on that
feature ID.
```

But we must not break native semantics:

```text
A protected consumed feature must not be fed back into MSCKF or SLAM update just
because it is retained for map capture.
```

## Current Minimal Database Decision

We tried adding generic history fields to `Feature` and SLG-specific descriptor
history in a `FeatureDatabaseSLG`. That was too much change before proving the
minimal protected-cleanup idea was insufficient.

The current decision is deliberately smaller:

```text
Protected features may stay in FeatureDatabase with to_delete still set.
to_delete continues to mean "do not use this feature as a fresh measurement".
No additional Feature fields are added yet.
```

## FeatureDatabase Semantics

`FeatureDatabase::cleanup(protected_ids)`:

- erases normal `to_delete` features
- retains protected `to_delete` features
- does not clear `to_delete`

`FeatureDatabase::cleanup_measurements(timestamp, protected_ids)`:

- trims old active measurements for normal features
- skips protected features so their current stored track remains intact until we
  understand exactly what death-time capture needs

`FeatureDatabase::update_feature(...)`:

- if the feature is not present, creates a new `Feature`
- if the feature is present and `to_delete == false`, appends normally
- if the feature is present and `to_delete == true`, it starts a fresh active
  measurement batch by clearing `uvs`, `uvs_norm`, and `timestamps`, then clears
  `to_delete` and appends the new observation

This models native delete/recreate behavior without actually erasing a protected
feature object.

## Why to_delete Must Stay Meaningful

Clearing `to_delete` during cleanup is wrong. It allows consumed measurements to
look active again and can feed stale information into the filter.

Keeping `to_delete` but not checking it is also wrong. A retained feature pointer
can still be found by `get_feature()`, so every place that asks "is this feature
currently observed?" must treat:

```cpp
feat != nullptr && !feat->to_delete
```

as the real current-observation test.

This is especially important when current in-state SLAM landmarks are gathered
for `UpdaterSLAM`. Protected retained tracks are map-capture material, not live
measurements.

## Descriptor History Is Deferred

SLG will eventually need SuperPoint descriptor history. KLT will not. ORB-like
future mappers may need their own metadata.

For now, no descriptor storage is added. We first test whether protected
`to_delete` retention alone preserves native estimator behavior. If it still
fails, we will locate the exact missing distinction and then decide whether
additional fields, subclassed features, or a separate archive are justified.

## UpdaterSLGM Timing

`UpdaterSLGM` must run after the system knows which EKF SLAM landmarks are about
to be marginalized, but before those landmarks are removed from the state and
before protected feature history is allowed to disappear.

Desired lifecycle:

```text
1. TrackSLG updates FeatureDatabase
2. OpenVINS promotes sufficiently exposed tracks to EKF SLAM landmarks
3. While EKF landmark is alive:
     its source feature ID is protected during database cleanup
4. EKF landmark is marked for marginalization
5. UpdaterSLGM processes the dying landmark:
     - reads landmark 3D position
     - reads the retained Feature by featid
     - matches to existing persistent map or spawns a new map landmark
6. EKF landmark is marginalized
7. feature protection is removed; later cleanup may erase the retained feature
```

The exact hook point is still to be implemented. It likely belongs near existing
SLAM marginalization logic in `VioManager`/`UpdaterSLAM`, before
`StateHelper::marginalize_slam(state)` removes the landmark from the EKF state.

## Persistent Map Landmark

Suggested minimal v1 structure:

```cpp
struct SLGMDescriptor {
  cv::Mat descriptor;
  size_t hits = 0;
  size_t attempts = 0;
  double first_seen = -1.0;
  double last_seen = -1.0;
};

struct SLGMLandmark {
  size_t map_id;
  Eigen::Vector3d p_G;
  std::vector<SLGMDescriptor> descriptors;

  size_t match_hits = 0;
  size_t match_attempts = 0;
  double first_seen = -1.0;
  double last_seen = -1.0;
};
```

Observation bearings were discussed but are not required for v1. They may become
useful later for sparse relocalization or descriptor disambiguation.

## Descriptor Dictionary Culling

Each persistent landmark owns a descriptor dictionary. The same 3D point may have
multiple descriptors from different views or lighting conditions.

When the descriptor cap is reached, cull by usage:

```text
erase descriptors with 0 hits
then descriptors with 1 hit
then descriptors with 2 hits
...
until under cap
```

This intentionally avoids premature manual deduplication. If duplicate map
landmarks exist, unused descriptor entries should eventually fall out once decay
is added.

V1 may omit decay. V2 should probably decay hit counts or maintain a recent-use
score so naturally duplicated map landmarks can die without active merging.

## Matching Hypothesis

Persistent landmarks and current EKF landmarks are both 3D point clouds.

LightGlue is fundamentally a 2D matcher, so the simple v1 policy is:

```text
project map landmarks into the current camera view
project current EKF landmarks into the current camera view
run LightGlue on those 2D projected keypoints and descriptors
accept robust matches
```

We discussed multi-projection matching as a future robustness check:

- project to multiple orthogonal planes
- run descriptor matching in each projection
- keep assignments that persist across projections

This is out of scope for v1. Normal camera projection is the KISS path.

## Duplicate Policy

Do not actively merge mature map landmarks in v1.

Duplicates are acceptable if they were born naturally. Active geometric merging
can easily damage the map. The preferred long-term duplicate control is
descriptor/list culling plus decay.

One exception was discussed for the rejected growing-landmark design: a mistaken
growing landmark could be merged into a mature one. Since v1 has no growing
landmarks, this exception does not apply.

## UpdaterKLTM

`UpdaterKLTM` is the descriptor-less sibling idea.

It can use the same protected `FeatureDatabase` retention rule:

- protected source features remain available until EKF landmark death
- no SLG descriptor dictionary is needed
- association must be spatial/temporal/geometric rather than descriptor-based

This is the main reason the first experiment should live in `FeatureDatabase`
rather than in an SLG-only sidecar archive.

## Sharp Edges

- Any "currently observed" feature query must ignore retained `to_delete`
  features.
- Protected cleanup must not clear `to_delete`.
- New observations of a protected consumed feature must begin a fresh active
  measurement batch.
- Protected retention should not alter the measurements passed into MSCKF or
  SLAM updates.
- Dying EKF landmarks must be processed before their feature IDs stop being
  protected.
- ArUco landmarks are a separate subsystem and should not be mixed with SLGM
  feature protection unless explicitly intended.

## Current Implementation Status

- `FeatureDatabase` supports protected cleanup.
- `FeatureDatabase::update_feature(...)` restores native active-batch semantics
  for reobserved consumed features.
- `VioManager` protects source feature IDs of alive non-ArUco EKF SLAM
  landmarks during feature cleanup.
- `UpdaterSLGM` itself is still future work.
