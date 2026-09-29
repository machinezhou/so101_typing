# SO-101 Closed-Loop Typing

A hybrid learned/classical robotic typing system for the SO-101 arm,
combining **ACT-based coarse motion**, **appearance-based key
recognition**, **wrist-camera visual servoing**, and **independent
screen-based verification**.

The project is designed as a compact embodied-AI system in which learned
policies and deterministic algorithms have explicit responsibilities,
explicit interfaces, and observable handoff points.

> **Core principle:** solve deterministic problems deterministically,
> and use learning where learning provides real value.

------------------------------------------------------------------------

## Project Goal

The goal is **not** to build the fastest robotic typist and **not** to
hide the task behind a pre-programmed keyboard map.

The goal is to build a reliable and interpretable physical closed loop
that can:

1.  receive a target string such as `ROBOT`,
2.  choose the next requested key,
3.  use ACT to move the SO-101 into a safe local neighborhood where the
    requested key is clearly visible to WRIST,
4.  visually recognize the requested key from its actual appearance,
5.  hand control from ACT to the deterministic wrist-vision controller,
6.  align the detected target-key center with the calibrated WRIST pencil-tip
    pixel reference,
7.  once alignment is stable, advance through small cumulative Z command
    increments relative to the same fixed command-space anchor, stopping after
    each commanded motion to reacquire WRIST and realign XY at the same Z when
    needed,
8.  keep an independent SIDE screen-change watcher active during the press so a
    persistent new continuation glyph can preempt any further descent/XY chase,
9.  on a latched SIDE event, release the key first with a short staged upward Z
    move, then classify the released character independently,
10. use screen evidence to choose SUCCESS / WRONG / UNCERTAIN and execute a
    bounded staged retract without treating commanded Z depth as success,
11. recover from wrong/uncertain/bounded failures and repeat until the requested
    string is correct.
    A typical final task is:

``` text
TARGET: ROBOT
```

and the robot should physically produce:

``` text
ROBOT
```

on the MacBook using only physical keyboard interaction.

------------------------------------------------------------------------

## Current Project Status

Phases 0–6 are accepted. The deterministic local single-key loop passed cross-key
hardware acceptance on `G`, `F`, and `H` without per-key tuning of `p_tip`,
`J_cmd`, SIDE thresholds, tracking, OCR, or press/release behavior. Phase 6 then
completed the data-qualification bridge required before ACT training.

Phase 7 is complete. The formal `keyboard_v1` ACT checkpoint was trained to
20k steps and accepted on real hardware with the same `n_action_steps=20` runtime
across `G`, `F`, `Q`, `P`, and `M`. Every retained PASS reached real deterministic
WRIST takeover and automatic HOME recovery. The moving `<=80 px` condition remains
only a **handoff candidate trigger**; deterministic WRIST convergence remains the
takeover ground truth.

`Z` was exercised twice with the same retained runtime. Both runs ended in
`TIMEOUT_NO_TAKEOVER`: saved WRIST evidence shows that the physical ACT approach
reached the Z neighborhood, but the runtime WRIST glyph recognizer produced no
accepted `Z` observation. This is retained as a known target-perception limitation,
not classified as an ACT coarse-policy failure, and no Z-specific ACT tuning is
introduced to force a pass.

**Phase 8 — ACT + Visual Servo Handoff — has begun.** The first full `G` integration
pilot physically completed ACT approach → deterministic WRIST control → staged Z
descent → SIDE event → immediate release → retract → HOME. The software did not
accept the run as end-to-end SUCCESS because the released-character OCR classified
the visibly correct lowercase `g` as `Q` on all 5 verification frames. Phase 8 is
therefore in integration-hardening / acceptance work, not completed.

Phase 6 established a generic episode pipeline rather than a task-specific
collector. Start-state coverage is an **external collection strategy/SOP**; the
program does not contain `HOME`, `LEFT`, `RIGHT`, `FRONT`, or `RETRACT-LIKE`
semantics and does not use those labels to alter collection, QC, or replay.

The completed Phase 6 funnel is:

``` text
natural human teleoperation episode
        ↓
generic collector
        ↓
automatic offline episode QC
        ↓
QC-PASS candidate episode
        ↓
full-rate physical replay of recorded actual sent actions
        ↓
seamless handoff at the recorded episode end
        ↓
real deterministic WRIST-servo takeover
        ├── fail → exclude episode
        └── pass → verified ACT episode
        ↓
verified episode manifest for Phase 7
```

The final Phase 6 contracts are:

- the **whole episode** is the unit of collection, QC, replay validation, and
  acceptance;
- formal collection uses natural expert trajectories with no `d_tip` gate,
  proximity radar, visual coaching, or automatic endpoint decision;
- the operator marks the natural end of the coarse-approach demonstration;
- ACT training samples are stored at 15 Hz, while a separate approximately 60 Hz
  actual-sent-action trace preserves command history for physical replay;
- observations are TOP RGB + WRIST RGB + six Present joint positions + 28D target
  one-hot; SIDE/SCREEN pixels are excluded from the initial ACT observation;
- the action stored for ACT is the actual absolute joint-position goal returned by
  `robot.send_action(...)`, not merely the requested action;
- offline QC removes clear data/trajectory defects but does not claim servo
  readiness;
- takeover ground truth is **full episode replay followed by the real deterministic
  WRIST servo**, not a visual-distance threshold or a final-pose teleport;
- Phase 6 acceptance stops after stable WRIST XY convergence. Z descent, SIDE
  press detection, release, OCR, and retract remain Phase 5/Phase 8 concerns and
  are intentionally excluded from per-episode Phase 6 acceptance;
- replay validation runs candidate episodes as a batch: there is no HOME reset
  between episodes, and the robot returns once at the end to the explicit recovery
  pose in `configs/robot/recovery_home.json`;
- safe transport may initialize from `Present_Position`, but deterministic WRIST
  control still preserves the existing `Goal_Position` as its fixed command-space
  anchor after takeover;
- only replay/takeover-verified episode indices are eligible for Phase 7 training.

### Formal Phase 6 target validation workflow

Formal validation is **target-scoped**, not latest-run-scoped. One target may
span multiple collection runs because collection can be interrupted by hardware
communication errors or because failed source zones are recollected later. Raw
collection remains append-only.

The normal validation command is:

``` bash
python scripts/phase6_validate_target.py --target U
```

This single command performs:

``` text
all accepted U episodes across every run_*_u
        ↓
target-level automatic QC
        ↓
default-scheduled QC-PASS episodes
        ↓
one continuous physical replay + WRIST takeover batch
        ↓
no HOME reset between episodes
        ↓
one final recovery HOME
        ↓
refresh verified_episodes.json
```

Target-level aggregation and automatic replay scheduling are separate concerns.
Aggregation always keeps the complete accepted raw history for the target.
Automatic scheduling uses the latest meaningful validation history for each
QC-PASS episode:

| Episode validation state | Default automatic replay | Meaning |
|---|---:|---|
| no validation trial | yes | needs first physical validation |
| `ABORTED` / `UNKNOWN` | yes | validation did not reach a terminal result |
| `PASS` | no | verified ground-truth result already exists |
| `FAIL` | no | exclude this raw episode and recollect a replacement under the external SOP |
| `INVALID_REPLAY` | no | replay itself was invalid; keep the raw episode for audit and recollect a replacement under the external SOP |

`INVALID_REPLAY` is therefore **not** a verified result and is **not** equivalent
to `PASS` or `FAIL` ground truth. It is terminal only for the *default automatic
scheduler*, so an old invalid replay cannot be pulled into every later
supplemental batch forever. The verified manifest policy is unchanged: an
episode becomes training-eligible only after the required physical takeover
`PASS` evidence (and no recorded takeover `FAIL`).

An operator may deliberately repeat one old QC-PASS episode with:

``` bash
python scripts/phase6_replay_takeover_validate.py --target U --episode <dataset_episode_index>
```

`--all` remains an explicit debug/repeat override and is not the normal
collection workflow.

The source-zone distribution remains an **external SOP**. The collector,
target-level QC, and validator do not encode S0-S5/start-class semantics.
Therefore dataset episode numbers, QC output order, and the final physical
`FAILED EPISODES` list must not be used by software to infer which source zone
still needs replacement. In particular, a QC `REVIEW`/`FAIL` episode never
enters the physical replay batch, so the physical batch failure count is not a
complete count of missing external-SOP source coverage.

The lower-level commands remain available for debugging:

``` bash
python scripts/phase6_episode_qc.py --target U
python scripts/phase6_replay_takeover_validate.py --target U

python scripts/phase6_episode_qc.py --run-dir <specific-run>
python scripts/phase6_replay_takeover_validate.py --run-dir <specific-run>
```

`--target` means **all formal runs for that target**. `--run-dir` means exactly
one collection run.

Target-level QC artifacts are stored outside individual collection runs:

``` text
artifacts/phase6_act_dataset/keyboard_v1/target_qc/<target>/qc_report.json
artifacts/phase6_act_dataset/keyboard_v1/target_qc/<target>/qc_candidates.json
artifacts/phase6_act_dataset/keyboard_v1/target_qc/<target>/takeover_batch_summary.json
```

Initial WRIST semantic-lock diagnostics also use the actual requested target.
For example, target `U` reports `semantic U lock`; target `P` reports
`semantic P lock`. The error is no longer hard-coded to `G`.

When initial semantic lock fails, the diagnostic reports how many fresh WRIST
frames were checked, how many frames recognized the requested target, how many
frames contained at least four detected keycaps, and the final keycap count. It
also saves the last WRIST debug image as `initial_semantic_lock_failure.png`
plus its overlay. This distinguishes “requested glyph was never recognized”
from insufficient keycap geometry without changing any WRIST control threshold,
the 1 mm FK safety gate, or the 35 mm cumulative XY command budget.

Implementation files used by this workflow:

``` text
scripts/phase6_target_scope.py
scripts/phase6_validate_target.py
tests/test_phase6_target_scope.py
```

A temporary `G` pilot dataset was used to validate the complete Phase 6 → Phase 7
pipeline, including offline QC, full physical replay, deterministic WRIST takeover,
verified-only normalization, ACT forward/backward/optimizer execution, and
checkpoint serialization. That pilot dataset and its derived training artifacts
were intentionally deleted before formal `keyboard_v1` work. Its conclusions remain
valid as engineering evidence, but the current 20k ACT checkpoint is a later formal
`keyboard_v1` runtime artifact rather than that deleted pilot.

Phase 3 remains the deterministic command-space foundation, Phase 4 remains the
independent screen semantic verifier, and Phase 5 remains the accepted local
press primitive. Phase 7 has completed verified-only input contracts, temporal
validation, official ACT training/checkpoint execution, trained-policy hardware
rollout, and multi-target ACT-to-WRIST takeover acceptance. Active work is now
Phase-8 learned-to-deterministic integration hardening.

The accepted Phase 7 infrastructure contract is:

- formal A-Z demonstrations append to one shared LeRobot dataset under
  `artifacts/phase6_act_dataset/keyboard_v1`;
- training samples are selected only by the dataset-level verified-episode
  manifest (`phase6.verified_episode_manifest.v4`);
- rejected/review raw episodes may remain in the raw dataset for audit/debugging,
  but only verified episode indices are eligible for training;
- normalization statistics are recomputed from exactly the verified subset rather
  than reusing repository-wide `meta.stats`;
- verified episode count and verified frame count are dynamic contract outputs,
  not hard-coded pilot constants;
- ACT runs at 15 Hz with `chunk_size=20`, corresponding to a `1.267 s`
  first-to-last action span and `1.333 s` worth of 20 commanded actions;
- `n_action_steps` is intentionally **not frozen yet** and will be selected after
  trained-policy inference-latency/runtime characterization;
- the training entry point uses the official LeRobot 0.6.1 ACT model,
  preprocessors, optimizer preset, ResNet18 backbone, and ImageNet initialization,
  while the project-specific wrapper injects verified-only normalization stats;
- batch size `8` completed forward/backward/optimizer/checkpoint smoke validation
  on the current RTX 5070 Ti with approximately `3.19 GiB` peak CUDA allocation.

## Research Questions

The central research question is:

> **Can a learned coarse-motion policy and a classical closed-loop
> visual controller cooperate to perform reliable physical interaction,
> while an independent visual observer verifies the real-world result
> and drives recovery?**

The project also studies several more specific questions:

- Can ACT learn a **perception-aware approach pose** rather than needing
  millimeter-level final accuracy?
- Can classical vision recognize the requested key locally after ACT has
  reduced the search space?
- Can an image-based visual-servo controller reliably remove the
  residual positioning error of a low-cost arm?
- What is the best interface between a chunked learned policy and a
  high-frequency deterministic controller?
- Does independent screen verification substantially improve final task
  success?
- How do learned visual representations and explicit classical visual
  features complement each other?
- How much does visual servoing improve over ACT-only execution?

Primary controller comparison:

| System | Learned coarse motion | Appearance-based local vision | Closed-loop fine correction | External screen verification | Recovery |
|---|---:|---:|---:|---:|---:|
| Classical baseline | No | Yes | Yes | Yes | Yes |
| ACT only | Yes | Limited/diagnostic | No | Yes | Yes |
| ACT + Visual Servo | Yes | Yes | Yes | Yes | Yes |

Possible future comparisons include SmolVLA and other learned policies,
but they are not dependencies of V0.

------------------------------------------------------------------------

# Design Constraints

## 1. Key identity must come from visual appearance

A deliberate project constraint is:

> **The system must not identify a key solely from a pre-encoded
> keyboard layout or from its row/column position.**

For example, the system should not conclude that a key is `F` only
because it is the fourth key in a known row.

Keyboard geometry may still be used for:

- finding keycap candidates,
- rejecting implausible contours,
- grouping nearby keycaps,
- geometric normalization,
- estimating local orientation.

However, the final key identity must be supported by the visible
glyph/appearance of the key itself.

This keeps the perception problem real and makes the interaction between
learned and classical vision meaningful.

## 2. V0 is a fixed-environment system

The first reliable version intentionally assumes:

- one fixed MacBook,
- fixed laptop position,
- fixed screen angle,
- fixed robot base,
- fixed camera mounts,
- fixed lighting as far as practical,
- fixed wrist-camera/tool geometry during a run.

V0 is intended to establish a reliable closed loop before introducing
generalization.

## 3. Screen verification is external to ACT and authoritative for press success

The screen camera is not part of the initial ACT observation.

ACT should learn **how to approach a requested key**, not infer success
from the MacBook display.

During deterministic staged pressing, the supervisor uses the independent
screen observation in two layers: a fast persistent-change event can immediately
stop any further press progression and trigger key release, while OCR determines
which character actually appeared. Commanded Z depth alone is never treated as
proof of success.

## 4. Preserve command-space preload across deterministic control

Physical Phase 3 testing showed material dead zone/backlash/compliance in the
SO-101 command chain. This applies to X, Y, and Z: a small requested Cartesian
change can produce little or no immediate measured motion, while larger
cumulative command changes can cross the dead zone and produce repeatable
image-space motion.

The deterministic controller therefore uses the servo's **existing
`Goal_Position`** values at handoff as the fixed command-space anchor. The
current `Present_Position` is still valuable for diagnostics, motion-stability
checks, safety guards, and IK seeding, but it must not be repeatedly promoted to
a new command origin. Doing so discards the servo preload and can create a
visible handoff jump even for an intended zero Cartesian correction.

After the anchor is latched, XYZ commands are cumulative with respect to that
same Goal-space reference. A tiny command that appears not to move the arm is
not, by itself, evidence that the command direction or controller logic is
wrong.

------------------------------------------------------------------------

# Hardware Setup

## Robot

- SO-101 arm
- calibrated follower/leader system
- Feetech servos
- physical tool currently implemented as a pencil attached to the side
  of the gripper with multiple cable ties
- MacBook fixed in the workspace

The current pencil mount is sufficient for V0 because physical key
pressing has already been demonstrated.

A rigid/custom printed mount can be introduced later to improve
repeatability of the camera-to-tool transform, but it is not a
prerequisite for starting the software pipeline.

## Cameras

The current three-camera configuration is:

``` bash
--robot.cameras='{
  top: {
    type: opencv,
    index_or_path: 2,
    width: 640,
    height: 480,
    fps: 30,
    fourcc: "MJPG"
  },
  wrist: {
    type: opencv,
    index_or_path: 0,
    width: 640,
    height: 480,
    fps: 30,
    fourcc: "YUYV"
  },
  side: {
    type: opencv,
    index_or_path: 4,
    width: 640,
    height: 480,
    fps: 30,
    fourcc: "YUYV"
  }
}'
```

The logical name `side` is retained for compatibility with the existing
setup, but in this project its primary role becomes **screen
verification**.

### Physical camera layout

``` text
                         TOP CAMERA
                             │
                             ▼
                 robot + keyboard workspace
                             │
                             │
              ┌──────────────┴──────────────┐
              │                             │
           SO-101                        MacBook
              │                             │
          WRIST CAMERA                  display
              │                             ▲
              ▼                             │
       local keyboard                  SIDE CAMERA
       / tool region                  (~45° allowed)
```

The current physical layout is considered suitable for V0.

------------------------------------------------------------------------

# Camera Responsibilities

The cameras intentionally have asymmetric roles.

## TOP — global learned-motion context

The TOP camera should prioritize visibility of:

- the SO-101,
- the keyboard,
- the useful robot motion volume,
- the spatial relationship between robot and laptop.

It answers:

> **Where should the robot move globally?**

The TOP camera does **not** need to resolve individual glyphs.
The MacBook display may remain visible in the TOP image. In the current
geometry it is strongly overexposed, and physically hiding it is not
worth sacrificing robot-workspace coverage.

For experimental rigor, preprocessing should support an optional fixed
screen mask:

``` text
TOP raw
   │
   ├── raw mode ──────────────► ACT
   │
   └── screen-mask mode ──────► ACT
```

This enables a later leakage/ablation experiment without changing the
physical setup.
## WRIST — local perception and precision control

The WRIST camera is the primary precision sensor.

It is responsible for:

- observing a small neighborhood of keys,
- detecting keycap candidates,
- recognizing visible key glyphs,
- finding the requested target key,
- estimating the target center in image coordinates,
- determining whether the target is suitable for controller handoff,
- measuring image-space alignment error during visual servoing.

It answers:

> **Which key am I looking at, and how far is the tool from the desired
> alignment?**

The wrist camera is shared by ACT and classical perception, but the two
consumers use it differently:

``` text
WRIST RGB
   │
   ├──► ACT
   │      implicit visual representation
   │      coarse approach behavior
   │
   └──► Classical Perception
          explicit key candidates
          explicit glyph labels
          explicit pixel coordinates
```

## SIDE / SCREEN — independent outcome verification

The SIDE camera is positioned as close to the MacBook screen as the robot
workspace safely permits. A view around 45° is acceptable.

It answers two different questions with two different paths:

> **Did any new keypress result appear on the screen?**

and, after the tool has been released:

> **Which character did that physical action produce?**

Because the MacBook and camera are fixed, perspective distortion is removed with
one screen homography. Phase 5 now uses the rectified view in two layers:

``` text
SIDE raw frame
      ↓
fixed screen quadrilateral
      ↓
perspective rectification
      ↓
canonical screen + fixed typing ROI
      ├──► FAST EVENT PATH
      │      pre-press baseline learns normal cursor ON/OFF variation
      │      continuation-region foreground change
      │      strong glyph → latch on first changed frame
      │      ordinary change → require 2 consecutive frames
      │      → latch press event
      │      → forbid further descent / XY chase
      │      → release key first
      │
      └──► SEMANTIC VERIFICATION PATH
             Phase 4 line OCR for stable NO_CHANGE / WRONG / SUCCESS evidence
             + Phase 5 released-character single-glyph OCR after event release
```

The fast event path deliberately does **not** run Tesseract on every frame. Its
job is only to detect a persistent new foreground component quickly. OCR remains
the semantic classifier, not the event detector.

The SIDE/SCREEN stream remains outside the initial ACT observation.

# System Architecture

The primary architecture is now:

``` text
                         TARGET TEXT
                           "ROBOT"
                              │
                              ▼
                     ┌─────────────────┐
                     │ Task Supervisor │
                     └────────┬────────┘
                              │ target key
                              ▼
                         ACT APPROACH
                    TOP + WRIST + state
                              │
                       servo-ready pose
                              ▼
                    CONTROLLER HANDOFF
                 stop/clear ACT action chunk
                 wait until motion-stable
                 latch existing Goal_Position
                              │
                              ▼
                    WRIST LOCAL CONTROLLER
                semantic lock + visual-servo XY
                              │
                    small cumulative Z step
                              │
                  stop / settle / fresh WRIST
                              │
                same-Z XY realignment if needed
                              │
                              ├───────────────────────────────┐
                              │                               │
                              │                        SIDE WATCHER
                              │                  persistent new foreground?
                              │                               │ yes
                              │                               ▼
                              │                    PRESS EVENT LATCHED
                              │                     no more Z-down / XY
                              │                               │
                              │                    one upward release escape
                              │                   +20 commanded-mm (clamped at Z=0)
                              │                               │
                              │                    released-char OCR
                              │                               │
                              │                 SUCCESS / WRONG / UNCERTAIN
                              │                               │
                              └──────── no event ─────────────┤
                                                              ▼
                                                    staged retract / recovery
```

A conservative Phase 4 line-OCR verification pass is still useful at stopped,
aligned levels when no fast event has fired. It can confirm `NO_CHANGE` before
authorizing another descent step. Once the fast SIDE event is latched, the press
attempt becomes one-way: no later observation may authorize more downward
motion.

The important architectural property is **controller ownership and command-space
continuity**:

- ACT owns coarse motion only until a perception-ready handoff.
- The handoff waits for physical motion stability and preserves the existing
  servo `Goal_Position` as one immutable deterministic command-space anchor.
- The deterministic controller owns cumulative XY alignment and cumulative Z
  press/release/retract commands after handoff.
- WRIST is the authority for local target alignment; semantic identity is
  established visually and geometry may only propagate an already-established
  identity through temporary occlusion.
- SIDE is the autonomous press-outcome authority. Its fast event latch has
  priority over normal press-loop progression and permanently disables further
  descent for that attempt.
- Robot commands remain on the foreground control thread. The current event
  watcher preempts at control-loop checkpoints; it does not yet cancel a
  `send_action()` call that is already in flight.
- The supervisor is the only module that changes high-level task state.

# Runtime State Machine

The runtime is an explicit state machine rather than a loose sequence of
function calls.

``` text
IDLE
  ↓
SET_TARGET
  ↓
ACT_APPROACH
  ├── target not visible / not ready ─────► continue ACT
  ↓
TARGET_ACQUIRED
  ↓
HANDOFF
  ├── stop ACT / clear pending action chunk
  ├── wait until follower motion is stable
  ├── preserve existing Goal_Position as fixed command anchor
  └── acquire fresh WRIST frame
  ↓
SERVO_ALIGN
  ├── target lost / stale / safety failure ─► RECOVERY
  └── SIDE_EVENT at any checkpoint ─────────► PRESS_EVENT_LATCHED
  ↓
ALIGNED_AT_LEVEL
  ↓
DESCEND_STEP
  └── SIDE_EVENT ───────────────────────────► PRESS_EVENT_LATCHED
  ↓
SETTLE_AND_REOBSERVE
  ├── drifted ─► REALIGN_AFTER_STEP
  │                └── SIDE_EVENT ──────────► PRESS_EVENT_LATCHED
  └── aligned ─► VERIFY_LEVEL
                  ├── CONFIRMED_NO_CHANGE ─► DESCEND_STEP
                  ├── UNCERTAIN ───────────► HOLD / REOBSERVE
                  ├── CONFIRMED_WRONG ─────► RETRACT ─► RECOVERY
                  └── CONFIRMED_SUCCESS ───► RETRACT ─► NEXT_TARGET

PRESS_EVENT_LATCHED
  ├── permanently forbid more Z-down / XY chase for this attempt
  ↓
RELEASE_KEY
  └── one +20 commanded-mm upward Z escape
      (clamped at command-space Z=0)
  ↓
VERIFY_RELEASED_CHARACTER
  ├── expected char ─► RETRACT ─► SUCCESS / NEXT_TARGET
  ├── wrong char ────► RETRACT ─► RECOVERY
  └── uncertain ─────► bounded reobserve only; NEVER descend
```

`TARGET_ACQUIRED` and `ALIGNED_AT_LEVEL` remain intentionally different states.
Every Z-level change invalidates the previous WRIST alignment acceptance. The
controller must stop, settle, acquire fresh observations, and realign at the same
stopped Z level if necessary.

The fast SIDE event adds a stronger invariant: **once a persistent screen change
is latched, the attempt may release, observe, retract, or recover, but it may
never descend again.** Commanded Z remains a command-space variable, not proof
of contact or success.

# Module Responsibilities and Interfaces

## Task Supervisor

The supervisor owns:

- target string,
- current expected character,
- task progress,
- controller transitions,
- verification interpretation,
- retry limits,
- recovery sequence,
- terminal success/failure.

Example:

``` text
target = "ROBOT"

R → O → B → O → T
```

The learned policy does not need to understand the word `ROBOT`. It
receives one requested primitive at a time.

## Target Encoding

Initial supported actions:

``` text
A-Z
SPACE
BACKSPACE
```

A simple symbolic encoding is sufficient for V0, for example:

``` text
A         → 0
B         → 1
...
Z         → 25
SPACE     → 26
BACKSPACE → 27
```

The implementation may use one-hot or another low-dimensional encoding.

The important requirement is that the target condition is explicitly
recorded in the dataset and explicitly supplied to the policy at
inference time.

## ACT Coarse Controller

ACT is responsible for:

> **Move the robot from a valid initial configuration into a viewpoint
> from which the requested key can be reliably acquired by the wrist
> perception system.**

This is intentionally different from:

> move the pencil tip perfectly to the key center.

Recommended ACT observations:

``` text
TOP RGB
WRIST RGB
robot joint state
target-key condition
```

Recommended action:

``` text
SO-101 joint-position command / action chunk
```

The existing joint-space ACT path should be retained initially because
the platform has already been demonstrated with ACT.

### Perception-aware approach

A successful ACT terminal state should make the next deterministic stage
observable and safe:

- requested key is visible,
- glyph is recognizable,
- key is sufficiently large in the wrist image,
- target is not heavily occluded by the pencil or arm,
- target has enough image-boundary margin for closed-loop correction,
- end effector remains at a safe pre-press height.

This is a **perception-aware approach pose**. ACT is not required to precisely
align the pencil with the key; removing the remaining image-space error is the
job of the deterministic WRIST controller.

## ACT → Visual Servo Handoff

The handoff must be triggered by perception, not by a fixed time or a fixed
number of ACT steps.

A conceptual observation is:

``` python
TargetObservation(
    target="R",
    detected=True,
    bbox=(x, y, w, h),
    center=(u, v),
    class_confidence=0.96,
    keycap_quality=0.93,
    occluded=False,
    stable=True,
    servo_ready=True,
)
```

For the future Phase 8 runtime, a `servo_ready` rule may combine conditions such
as:

``` text
correct target label
AND confidence >= threshold
AND key size >= threshold
AND complete usable target observation
AND enough image-boundary margin for correction
AND stable for N consecutive fresh frames
```

This is a runtime handoff contract to be validated empirically. **Phase 6 formal
demonstration collection must not use these conditions as operator guidance or as
an automatic endpoint gate.** Phase 6 retains the raw WRIST observations and
timing diagnostics needed for later analysis; handoff-related perception metrics
may be computed offline when needed. Handoff ground truth is established later
through full-episode replay plus real deterministic takeover.

Handoff does **not** require millimetre-level ACT placement or a sub-20-pixel
residual. Phase 3 hardware validation demonstrated successful deterministic
capture from an initial WRIST error of approximately **61.9 px**, reducing it
to approximately **4.6 px**. This is evidence for the current fixed setup, not
a guaranteed universal capture boundary.

When `servo_ready` becomes true:

1.  stop ACT and clear/reset any pending ACT action chunk,
2.  prevent any stale ACT action from reaching the robot,
3.  keep the follower/leader hardware session connected; do not reconnect the
    follower at the hover pose,
4.  wait until follower motion is stable,
5.  read and preserve the servo's existing `Goal_Position` values as the fixed
    deterministic command-space anchor,
6.  read `Present_Position` separately for diagnostics, safety checks, and IK
    seeding; do **not** use it to redefine the command origin,
7.  acquire a fresh WRIST frame after the stable handoff,
8.  transfer exclusive command ownership to the deterministic staged
    visual-servo/press controller.

This boundary is critical. ACT and the deterministic controller must never
command the robot concurrently in V0. Preserving the existing Goal-space preload
is equally important: hardware tests showed that rebasing a zero Cartesian
command on `Present_Position` can move the arm, whereas resending the unchanged
existing `Goal_Position` produced no visible or joint motion.

------------------------------------------------------------------------

# Wrist Key Perception

The wrist perception problem is intentionally local.

ACT reduces:

``` text
search the entire keyboard for R
```

to:

``` text
inspect a small local neighborhood and identify R
```

This is an important interaction between learning and classical
perception: the learned policy actively creates an easier perception
problem.

## Proposed pipeline

``` text
WRIST YUYV frame
       ↓
grayscale / illumination normalization
       ↓
dark keycap segmentation
       ↓
morphology
       ↓
contours / connected components
       ↓
geometric keycap filtering
       ↓
quadrilateral fitting
       ↓
per-key perspective normalization
       ↓
canonical key crop
       ↓
glyph recognition
       ↓
label + confidence + center
```

### Keycap detection

V0 should first attempt classical methods:

- grayscale/intensity segmentation,
- adaptive or fixed thresholding,
- morphology,
- contour extraction,
- area/aspect-ratio filtering,
- quadrilateral/rounded-rectangle geometry,
- local consistency checks.

A large object detector is not the default starting point because the
current fixed MacBook scene has strong keycap/background contrast.

### Glyph recognition

The project should compare at least:

1.  template matching,
2.  HOG + linear SVM,
3.  a small CNN.

Recommended classification space:

``` text
A-Z + OTHER
```

`OTHER` prevents number keys and modifier keys from being forcibly
classified as letters.

For `SPACE` and `BACKSPACE`, dedicated visual handling may be required
because they are not ordinary single-letter glyphs.

### Important constraint

Perspective rectification of an individual key is allowed.

Pre-programmed row/column identity is not.

The classifier must infer identity from visible appearance.

------------------------------------------------------------------------

# Visual Servo

After ACT/manual handoff, WRIST vision owns fine alignment. The accepted V0
primitive uses one fixed command-space anchor and a staged loop:

``` text
XY ALIGNMENT
    ↓
ALIGNED_AT_LEVEL
    ↓
ADVANCE ONE CUMULATIVE Z LEVEL
(hold cumulative XY fixed)
    ↓
STOP + SETTLE + REOBSERVE
    ↓
REALIGN XY AT FIXED Z IF NEEDED
```

A Z-level transition and a lateral correction are separate stopped stages. The
controller does not change XY and Z simultaneously as a corrective action.

## Tool-tip reference point

The control reference is the physical pencil-tip projection in the rigid WRIST
camera/tool image:

``` text
p_tip = (u_tip, v_tip)
```

The direct WRIST tool calibration is now complete and stored in
`calibration/tool_reference.json`. The accepted reference is approximately:

``` text
p_tip = (307.0, 238.246) px
```

The calibration used five direct samples with a maximum radial deviation of
approximately 1.62 px. The old H3.1 target-derived value remains historical and
must not be relabelled as the direct tool reference.

For a detected target-key center:

``` text
p_key = (u_key, v_key)
```

the image error is:

``` text
e = p_key - p_tip
```

The controller drives `||e||` toward zero. The same `p_tip` is shared across
ordinary A-Z keys; there is no per-key tool reference.

## Semantic lock and occlusion tracking

Target identity must still be established from visible glyph appearance. During
closed-loop motion, however, the pencil can occlude the glyph as it approaches
the target. Phase 3 therefore uses a two-stage visual contract:

``` text
glyph recognition establishes target identity
        ↓
short-range closed-loop motion
        ↓
if the glyph becomes hidden:
whole-keyboard keycap geometry estimates image translation
        ↓
carry forward the already-established target center
```

The geometry fallback is not allowed to choose a new key identity. It only
propagates an already-established semantic lock, and only when multi-key matching
passes strict inlier/residual gates. A single poor geometry frame is retried
within the fresh-frame timeout rather than immediately terminating the run.

## Local image Jacobian

For V0, the mapping from a bounded **requested Cartesian XY command** to WRIST
image motion is the accepted command-space Jacobian:

``` text
delta_p_image ~= J_cmd @ delta_c_requested
```

where `J_cmd` has units `px / commanded-mm`. The canonical calibration remains
`calibration/image_jacobian.json`:

``` text
J_cmd =
[[-2.386845, -0.414119],
 [ 0.294672,  2.875000]]
```

The millimetre value is the command-space scaling used by the LeRobot Cartesian
processor. It is not an independently measured TCP displacement or an SO-101
positioning-accuracy claim.

A local correction uses a damped/bounded form of:

``` text
delta_c_requested = -J_cmd^+ @ e
```

with bounded step size, cumulative command budget, fresh-frame requirements,
target-loss handling, and convergence checks.

## Fixed Goal-space anchor and dead-zone-aware accumulation

Phase 3 hardware testing showed that the SO-101 can remain motion-stable with a
non-zero `Goal_Position - Present_Position` residual. XYZ therefore must not be
controlled by repeatedly rebasing on `Present_Position`.

The accepted runtime contract is:

``` text
wait until handoff motion is stable
        ↓
read existing servo Goal_Position once
        ↓
latch it as the fixed command-space anchor
        ↓
apply cumulative XY / Z commands from that same anchor
        ↓
use Present_Position / FK only for diagnostics and safety
        ↓
use fresh visual outcome for XY success
```

Small requested Cartesian changes can be absorbed by dead zone/backlash. The
controller therefore accumulates command-space correction rather than treating a
small no-motion result as a direction failure or relatching a new origin. The
accepted H3.2 Jacobian was calibrated with 5 mm command perturbations; Phase 3
closed-loop validation likewise required bounded cumulative commands large
enough to cross the hardware dead zone.

## Alignment acceptance

Do not declare alignment from one frame. The current Phase 3 hardware validation
used approximately:

``` text
initial XY acceptance:  ||e|| <= 4 px for 2 consecutive fresh frames
Z-level handoff/recheck: ||e|| <= 6 px
```

These are validated V0 operating values, not claims of sub-pixel mechanical
accuracy. Near the target, image/geometry noise and backlash can create a few
pixels of chatter, so the controller should stop correcting inside an accepted
deadband rather than chase 1–2 px indefinitely.

Two representative hardware results are:

``` text
61.89 px initial error  -> 4.56 px after bounded cumulative XY control
43.43 px initial error  -> 1.79 px stable (2/2 fresh frames)
```

The demonstrated capture range is therefore at least about 62 px in the tested
fixed setup; ACT should be trained to create a safe, visible servo-ready state,
not to achieve the final pixel alignment itself.

------------------------------------------------------------------------

# Deterministic Press Controller

The press stage is deterministic in V0. It uses one fixed Goal-space anchor,
small cumulative Z steps, stopped WRIST re-observation, SIDE event detection,
and independent result verification.

Current supervised Phase 5 behavior is:

``` text
stable WRIST alignment
        ↓
small cumulative Z step (2 commanded-mm)
        ↓
wait until motion-stable
        ↓
fresh WRIST observation
        ↓
realign XY at the same Z if needed
        ↓
if no fast SIDE event:
    conservative line OCR may confirm NO_CHANGE
    → authorize one more 2 mm Z step

at any control-loop checkpoint:
    persistent SIDE continuation change (2 frames)
        ↓
    PRESS_EVENT_LATCHED
        ↓
    forbid all further Z-down and XY chase
        ↓
    release +20 commanded-mm once
        ↓
    classify released character
        ↓
    staged retract to command-space Z=0
```

The current supervised hardware validator deliberately uses
`max_descent_mm=None`. There is no arbitrary cumulative command-space Z cap in
the accepted supervised experiment because earlier hard depth levels incorrectly
assumed ACT/manual placement had already brought the tool close to contact. The
operator therefore remains the absolute-depth safety authority during this
checkpoint and uses Ctrl+C if the physical pose becomes unsafe. A future fully
autonomous runtime must add an absolute workspace/contact safety mechanism that
does not reintroduce the old assumption.

Other active safety/control limits remain explicit:

- press XY correction is bounded per step and by cumulative spatial budget,
- normal press planning keeps the strict 1.0 mm XY FK model-consistency gate,
- event release uses the latest actually-sent XY state and a bounded local FK gate,
- full retract uses a local 3.0 mm XY FK gate,
- release is currently one +20 commanded-mm upward escape, clamped at Z=0,
- deep retract is segmented into at most 10 commanded-mm Z changes so the
  LeRobot EE-jump guard is not violated,
- joint/workspace/clipping checks remain active,
- `Present_Position` and FK remain diagnostics/safety/IK-seed signals rather
  than press-success authority.

The current event/retract-specific gates do not relax normal press alignment.
They exist because upward escape motion has a different objective from precise
key targeting.

A completed motion command does **not** mean the arm is physically settled. The
SO-101 can retain sizeable Goal-vs-Present residuals because of preload,
backlash, compliance, and servo limits. The deterministic command state remains
cumulative from the original Goal anchor throughout the attempt.

# Screen Perception and Verification

The SIDE camera is an independent observer and now has two complementary
responsibilities.

## Stable semantic verifier — Phase 4 authority

The Phase 4 path remains accepted for stable screen interpretation:

``` text
SIDE frame
  ↓
rectify with calibration/screen_homography.json
  ↓
fixed typing ROI
  ↓
dark text-line localization
  ↓
3x crop + CLAHE
  ↓
Tesseract 5.3.4 / OEM 1 / PSM 7
whole-line whitelist = A-Z + 0-9
  ↓
confirmed-prefix + expected-character comparison
  ↓
9-frame vote
  ↓
CONFIRMED_SUCCESS / CONFIRMED_NO_CHANGE /
CONFIRMED_WRONG / UNCERTAIN
```

The whole-line whitelist intentionally remains uppercase-only. A Phase 5 hardware
attempt showed why: allowing `a-z` across the full line caused low-confidence
noise to be decoded as long runs of `g/a/q`, destroying the otherwise stable
`KEYPRESS` prefix. The verifier already normalizes case semantically, so the
solution is **not** to broaden the whole-line OCR alphabet.

## Fast press-event detector — Phase 5 extension

A press event should be detected faster than a full OCR vote. Before autonomy
starts, the SIDE watcher records a short baseline (currently about 1.4 s) so
normal cursor ON/OFF states are represented. During pressing it watches only the
continuation region after the confirmed prefix.

``` text
baseline frames (cursor ON/OFF included)
        ↓
current rectified continuation ROI
        ↓
foreground difference vs best matching baseline state
        ↓
connected-component / novel-pixel gates
        ↓
persistent for 2 consecutive frames
        ↓
PRESS_EVENT_LATCHED
```

The detector asks only **"did a persistent new glyph-like foreground appear?"**
It does not try to recognize the character and therefore does not need to run
Tesseract at camera rate.

## Released-character verification

After a press event is latched, the controller first releases the key. Only then
is the new continuation component classified. The single-character OCR path may
accept lowercase letters because the physical MacBook input can display lowercase
`g` even though the semantic target is `G`; the result is normalized before
comparison.

The first successful hardware run produced:

``` text
fast event: 759 novel pixels, one 29x49 component, 2 consecutive frames
release:    one +20 commanded-mm upward escape from the latched press state
char OCR:   G, G, G, G, Q
result:     CONFIRMED_SUCCESS (4/5 success votes)
```

This separates three concerns cleanly:

- **fast screen change** decides when to release,
- **OCR** decides which character appeared,
- **the supervisor** decides success/wrong/uncertain and recovery.

`UNCERTAIN` never authorizes deeper descent. Once the fast press event is latched,
no later observation can return the attempt to the descent path.

# Automatic Recovery

Recovery semantics belong to the supervisor.

Example:

``` text
target   = ROBOT
observed = ROBOR
```

Longest correct prefix:

``` text
ROBO
```

Recovery plan:

``` text
BACKSPACE
T
```

Another example:

``` text
target   = ROBOT
observed = ROXX
```

Longest correct prefix:

``` text
RO
```

Recovery plan:

``` text
BACKSPACE
BACKSPACE
B
O
T
```

Division of responsibility:

``` text
Screen Perception
"What happened?"

Supervisor
"What should happen next?"
ACT
"How do I move into a useful neighborhood?"

Wrist Perception
"Which visible key is the requested key?"

Visual Servo
"How do I remove the remaining local alignment error?"

Press Controller
"How do I execute the physical press safely?"
```

------------------------------------------------------------------------

# Camera Calibration and Sanity Checks

Before collecting the typing dataset, freeze the camera positions and
validate all three roles.

## Camera position acceptance

### TOP

Pass if:

- keyboard remains visible,
- useful robot workspace remains visible,
- normal arm elevation does not destroy the majority of useful context,
- the camera mount is mechanically stable.

The screen does not need to be physically removed from view.

### WRIST

Pass if:

- near an intended handoff pose, several local keys are visible,
- glyphs contain enough pixels for classification,
- the pencil/arm does not consistently hide the requested key,
- the camera-to-tool relationship remains stable during a run.

### SIDE / SCREEN

Pass if:

- the robot never collides with or requires the screen camera’s space,
- the relevant screen region remains visible,
- text remains recoverable after perspective rectification,
- exposure can be configured for readable screen text.

## Exposure and focus

Where supported, prefer stable settings for:

- exposure,
- gain,
- white balance,
- focus.

Each camera should be optimized for its own role rather than forced to
share identical settings.

- TOP: robot + keyboard context
- WRIST: black keycaps + white glyph detail
- SIDE: display text

## TOP screen-leakage test

Physical masking is not required.

For rigor, keep a software mask option and test whether the overexposed
TOP display still carries measurable information.

Example test states:

``` text
blank
R
RO
ROB
ROBOT
```

with the robot fixed.

Measure within the TOP screen ROI:

- saturated-pixel ratio,
- mean/std,
- frame-to-frame noise,
- inter-state pixel differences.
  If the states are indistinguishable from noise, raw TOP can be used with
  evidence that the screen carries no practical task signal.

If they remain distinguishable, enable the fixed software mask for ACT.

------------------------------------------------------------------------

# Timing, Freshness, and Controller Ownership

Visual servoing is more sensitive to latency than coarse ACT motion.

Every camera frame used by the runtime should carry:

``` text
camera_name
frame_id
capture_timestamp
processing_timestamp
frame_age_ms
```

Robot commands should carry:

``` text
command_timestamp
controller_owner
sequence_id
```

The servo controller should reject frames that are too old.

This prevents a failure mode such as:

``` text
robot moves
   ↓
controller processes an old image
   ↓
computes correction for the previous pose
   ↓
oscillation / wrong correction
```

At any instant, exactly one motion controller should own robot commands.

------------------------------------------------------------------------

# Recommended Runtime Data Contracts

These are conceptual contracts; exact implementation may use
dataclasses, Pydantic models, or typed dictionaries.

## `TargetObservation`

``` python
@dataclass
class TargetObservation:
    target: str
    detected: bool
    bbox: tuple[int, int, int, int] | None
    center: tuple[float, float] | None
    class_confidence: float
    keycap_quality: float
    occluded: bool
    stable: bool
    servo_ready: bool
    frame_id: int
    timestamp: float
```

## `AlignmentResult`

``` python
@dataclass
class AlignmentResult:
    converged: bool
    error_px: tuple[float, float]
    error_norm_px: float
    iterations: int
    elapsed_s: float
    target_confidence: float
    timeout: bool
```

## `PressResult`

``` python
@dataclass
class PressResult:
    completed: bool
    descent_steps: int
    cumulative_downward_command: float
    retracted: bool
    timeout: bool
    safety_abort: bool
```

## `VerificationResult`

``` python
class VerificationStatus(Enum):
    CONFIRMED_SUCCESS = "confirmed_success"
    CONFIRMED_NO_CHANGE = "confirmed_no_change"
    CONFIRMED_WRONG = "confirmed_wrong"
    UNCERTAIN = "uncertain"

@dataclass
class VerificationResult:
    status: VerificationStatus
    observed_text: str
    confidence: float
    frame_id: int
    timestamp: float
```

------------------------------------------------------------------------

# Event Logging and Replay

Every key attempt should produce a trace that can be inspected after
failure.

Example:

``` text
target=R
0.000  state=ACT_APPROACH
0.033  act_action=[...]
1.820  target=R confidence=0.91 center=(411,231)
1.821  transition=ACT_APPROACH->HANDOFF
1.830  act_queue=cleared
1.850  transition=HANDOFF->SERVO_ALIGN
1.870  servo_error=(+34,-19)
1.930  servo_error=(+22,-12)
2.040  servo_error=(+5,-3)
2.105  servo_error=(+1,+1)
2.205  transition=SERVO_ALIGN->DESCEND_STEP
2.260  fixed_goal_anchor=latched cumulative_xyz=(+6.8,+15.0,-3.0)mm
2.420  transition=SETTLE_AND_REOBSERVE->REALIGN_AFTER_STEP
2.500  alignment_error=(-7,+0)
2.700  cumulative_xyz=(+3.7,+15.4,-3.0)mm
2.840  transition=REALIGN_AFTER_STEP->VERIFY_LEVEL
2.980  verification=CONFIRMED_NO_CHANGE
2.981  transition=VERIFY_LEVEL->DESCEND_STEP
3.050  cumulative_xyz=(+3.7,+15.4,-6.0)mm
3.260  screen_text="R"
3.261  verification=CONFIRMED_SUCCESS
3.262  transition=VERIFY_LEVEL->RETRACT
```

Recommended logged signals:

- state transitions,
- target key,
- ACT actions,
- joint observations,
- frame IDs/timestamps,
- target detections,
- confidence values,
- servo errors,
- controller ownership,
- press commands,
- OCR output,
- recovery decisions,
- failure reason.

A replay/debug viewer is highly valuable for both engineering and the
final demonstration.

------------------------------------------------------------------------

# Dataset Design

## Wrist perception dataset

Before training a new ACT typing policy, collect a small dedicated wrist
dataset.

For each target region:

- move the robot near different letters through teleoperation,
- vary XY offset,
- vary safe pre-press height slightly,
- include moderate viewpoint variation,
- include partial occlusion,
- collect non-letter keys for `OTHER`.
  The first goal is not large-scale learning. It is to measure whether
  keycap detection and glyph recognition are reliable under the actual
  wrist-camera distribution.

Evaluate:

``` text
Template Matching
vs
HOG + Linear SVM
vs
Tiny CNN
```

using held-out frames/episodes rather than adjacent frames from the same
recording whenever possible.

## ACT typing dataset

After the local deterministic loop is reliable, collect target-conditioned
**natural coarse-approach episodes**.

Recommended policy observations:

``` text
observation.top
observation.wrist
robot joint state
target-key condition
```

Recommended action:

``` text
SO-101 joint-position command actually sent to the follower
```

Do not include the SIDE/SCREEN camera in the initial policy input.

The initial V0 target vocabulary is:

``` text
A-Z + SPACE + BACKSPACE
```

and the target condition must be explicitly stored in every episode.

### Natural demonstration endpoint

The formal collector must not force the operator to chase a hard-coded image
metric such as `d_tip <= 65 px`. Such a threshold may be useful during handoff
experiments, but using it as a collection constraint changes the human trajectory
and can contaminate the policy distribution.

Instead, one demonstration is:

``` text
valid NOT_READY start
        ↓
natural continuous leader teleoperation
        ↓
operator completes the intended coarse approach
        ↓
operator marks episode end
```

The collector should trim preparation/reaction tail, store the full natural
trajectory, and retain raw WRIST observations plus timing diagnostics without
running handoff perception online. Handoff-related perception metrics may be
computed offline when needed.

### Episode-level qualification

A raw collected episode is **not yet a training episode**. Qualification has two
stages.

First, offline episode QC rejects recordings with clear data or demonstration
problems, for example:

- missing/corrupt camera or robot data,
- invalid target conditioning,
- large unintended pauses or severe hesitation,
- repeated strong reversals,
- abnormal duration,
- significant requested-to-sent action clamp,
- unsafe or obviously pathological start/end states.

Offline QC may use WRIST metrics to describe the endpoint, but those metrics are
not ground truth for takeover success.

Second, every remaining candidate episode is validated physically:

``` text
restore/reconstruct the episode start
        ↓
replay the recorded actual sent-action sequence in order
        ↓
reach the recorded episode endpoint with its command history
        ↓
without resetting to Present_Position or teleporting to the final pose
        ↓
seamlessly transfer command ownership to the deterministic WRIST servo
        ↓
run the real takeover
        ├── FAIL    → reject the whole episode
        └── SUCCESS → accept the whole episode
```

Full-sequence replay is preferred over commanding only the final joint vector
because SO-101 backlash, dead zone, compliance, Goal/Present lag, and preload can
make endpoint behavior history-dependent.

### Final training-set contract

The Phase 6 training set is therefore:

``` text
natural episode
+ offline QC passed
+ full replay passed
+ deterministic WRIST-servo takeover passed
= verified ACT training episode
```

The **episode** is the acceptance unit. Individual frames remain necessary for
ACT observation/action training and later analysis, but Phase 6 does not define a
training example by searching each trajectory for an arbitrary “candidate
handoff frame.” The recorded episode endpoint is the intended handoff boundary;
physical replay/takeover validation decides whether that endpoint is acceptable.

For robustness, a later acceptance policy may require repeated replay/takeover
success for an episode (for example multiple successful trials). The exact repeat
criterion should be frozen only after empirical replay stability is measured.

------------------------------------------------------------------------

<!-- IMPLEMENTATION_PROGRESS:START -->

# Implementation Progress and Current Checkpoint

This section tracks the actual implementation and integration status of
the project. The Development Roadmap below is updated when hardware evidence
changes a phase boundary, command contract, or acceptance criterion.

## Overall Progress

| Phase    | Description                                         | Status |
|----------|-----------------------------------------------------|--------|
| Phase 0  | Mechanical Feasibility                              | **Completed** |
| Phase 1  | Freeze Camera Geometry and Build Camera Sanity Tool | **Completed** |
| Phase 2  | Wrist Keycap Detection and Glyph Recognition        | **Completed** |
| Phase 3  | Tool Reference and Visual Servo                     | **Completed** |
| Phase 4  | Screen Rectification and Verification               | **Completed** |
| Phase 5  | Deterministic Local Single-Key Closed Loop          | **Completed — G/F/H cross-key acceptance** |
| Phase 6  | Target-Conditioned ACT Dataset                      | **Completed — generic collection + QC + physical replay/takeover verification** |
| Phase 7  | ACT Coarse Policy                                   | **Completed — 20k keyboard_v1 checkpoint + multi-target ACT→WRIST takeover acceptance** |
| Phase 8  | ACT + Visual Servo Handoff                          | **IN PROGRESS — first G E2E pilot physically closed the loop; released-char OCR/software acceptance remains** |
| Phase 9  | Multi-Key Typing                                    | Not Started |
| Phase 10 | Automatic Recovery                                  | Not Started |
| Phase 11 | Controlled Generalization and Ablations             | Not Started |

A phase is complete only after its acceptance criteria have been validated on
the intended system.

## Current Phase

``` text
Phase 8 — ACT + Visual Servo Handoff
(integration hardening; Phase 7 accepted)
```

Current checkpoint:

``` text
Phase 1 camera/software foundation                   ✓ COMPLETED
        ↓
Phase 2 wrist perception                             ✓ COMPLETED
        ↓
Phase 3 fixed Goal anchor + WRIST visual servo       ✓ COMPLETED
        ↓
Phase 4 screen verification                          ✓ COMPLETED
        ↓
Phase 5 deterministic G/F/H single-key loop          ✓ COMPLETED
        ↓
Phase 6 generic collection + QC + replay/takeover    ✓ COMPLETED
        ↓
Phase 7 verified-only input + temporal contract      ✓ COMPLETED
        ↓
Phase 7 official ACT training/checkpoint path        ✓ COMPLETED
        ↓
keyboard_v1 ACT 20k checkpoint                       ✓ TRAINED
        ↓
Phase 7 rollout evaluator v6                         ✓ VALIDATED
        ↓
n_action_steps=20 G/G/F/Q/P/M takeover evidence      ✓ PASS
        ↓
Phase 7 multi-target ACT coarse-policy acceptance       ✓ COMPLETED
        ↓
Phase 8 first G full physical integration pilot      ✓ PHYSICAL LOOP CLOSED
        ↓
released-character software verdict                  ✗ BLOCKER: actual g -> OCR Q (5/5)
        ↓
remove validation-only runtime scaffolding           ← NEXT
        ↓
add autonomous absolute-depth/workspace safety bound ← NEXT
        ↓
fix released-char OCR from real run regression data  ← NEXT
        ↓
repeat G until software SUCCESS + HOME PASS           ← NEXT
        ↓
cross-key Phase 8 acceptance                         FUTURE
```

### Phase 7 runtime checkpoint

The retained Phase-7 trained-policy runtime baseline is:

``` text
checkpoint          = artifacts/phase7_act_coarse/keyboard_v1/act_train_20k/checkpoints/last
control rate        = 15 Hz
chunk_size          = 20
n_action_steps      = 20   # retained Phase-7 acceptance baseline
execution horizon   = 1.333 s
Temporal Ensemble   = OFF
```

The rollout evaluator deliberately separates three different quantities:

``` text
moving HANDOFF_CANDIDATE
        ↓
ACT stop + motion settle
        ↓
fresh SETTLED_ENDPOINT diagnostic
        ↓
independent deterministic WRIST takeover
        ↓
TAKEOVER PASS / FAIL = ground truth
```

The moving `<=80 px` condition is **not** called servo-ready and is not the
acceptance authority. It is only a candidate trigger. Final retained hardware
evidence is:

| Target/run | Moving candidate | Settled endpoint | Result |
|---|---:|---:|---|
| `G` v6 run 1 | `78.47 px` | `76.92 px` | PASS |
| `G` v6 run 2 | `78.34 px` | `72.40 px` | PASS |
| `F` v6 | `65.68 px` | `59.28 px` | PASS |
| `Q` v6 | `40.60 px` | `38.78 px` | PASS |
| `P` v6 | `75.16 px` | `72.25 px` | PASS |
| `M` v6 | `73.64 px` | `63.19 px` | PASS |

All retained PASS runs used the same 20k checkpoint, 15 Hz control rate, and
`n_action_steps=20`, with no per-target runtime tuning, and completed real
deterministic WRIST takeover plus automatic HOME recovery.

Two additional `Z` runs ended in `TIMEOUT_NO_TAKEOVER`. Saved WRIST evidence
shows that the physical ACT approach reached the Z neighborhood, while the runtime
glyph recognizer produced no accepted `Z` observation. The case is retained as a
known target-perception limitation and is not worked around with Z-specific ACT
parameters.

A diagnostic `n_action_steps=10` run exposed large direction discontinuities at
ACT chunk boundaries and eventually failed deterministic takeover. The existing
1 mm planner/FK sanity guard must not be loosened to make such a run pass.
`n_action_steps=20` is therefore retained as the Phase-7 acceptance baseline.

### First Phase 8 end-to-end pilot — 2026-09-29

The first `G` full integration run used the 20k ACT checkpoint and the current
`n_action_steps=20` baseline. The important physical sequence worked:

``` text
ACT approach
   ↓
handoff candidate / ACT stop
   ↓
deterministic controller owns the robot
   ↓
one fixed Goal_Position command anchor
   ↓
WRIST align + staged Z + same-Z realignment
   ↓
SIDE persistent change event
   ↓
release command sent 18.3 ms after event capture
   ↓
segmented retract to command-space Z=0
   ↓
automatic recovery HOME (final Goal diff ≈ 0.044 deg)
```

The SIDE event arrived while an inner WRIST command had already changed XY. The
outer state was stale, but `LatestSentXYZCommandState` correctly preserved the
actual most recently sent command and the release used that authoritative state.
This is direct end-to-end evidence that the earlier stale-event-state fix is
necessary and must be preserved.

The software result was still:

``` text
actual editor character  = g
single-char OCR votes    = Q / Q / Q / Q / Q
controller outcome       = WRONG_KEY
Phase 8 task status      = FAIL_PRESS_OUTCOME
HOME recovery            = PASS
```

The saved novel-character crop is visually complete; this is not currently
attributed to a broken SIDE crop. It is an unresolved single-character OCR
preprocessing/classification weakness. Phase 5 had already shown the same `g/Q`
ambiguity (`G / Q / G / G / Q`) but majority voting masked it. The first Phase 8
run turned the same weakness into a deterministic 5/5 `Q` result, so it must now
be fixed rather than tolerated by luck.

### Integration lessons that are now frozen

- A moving image-plane threshold is a **handoff candidate**, not servo-ready ground
  truth. Stop, settle, measure a fresh endpoint, then let actual WRIST convergence
  decide acceptance.
- ACT ownership must end before deterministic motion starts. No stale queued ACT
  command may execute after handoff.
- Deterministic alignment, Z descent, release, and retract must all remain cumulative
  from the **same fixed existing `Goal_Position` anchor**. Do not relatch from
  `Present_Position` during the local primitive.
- `Present_Position` / FK telemetry is diagnostic. Goal/Present lag under load is
  not evidence that fixed-anchor command semantics are wrong.
- Do not weaken the 1 mm planner/FK sanity guard or the 35 mm cumulative XY budget
  merely to make an ACT rollout succeed.
- A SIDE press event has priority over all later WRIST/Z work. Once latched, no more
  downward command or XY chase is permitted for that attempt.
- Event-time recovery must use the latest command that was **actually sent**, not a
  potentially stale outer Python state.
- Release first, classify second. OCR is the semantic success authority after
  release; it must never delay the safety release.
- Fixed `KEYPRESS` text and normal-path `ENTER` / `GO` prompts are Phase-4/5
  validation scaffolding, not formal Phase-8 runtime requirements.
- The current SIDE continuation model derives its continuation region from existing
  detectable text. A completely blank editor is therefore not yet a formally
  accepted start condition and needs an explicit design decision before the final
  typing demo.
- Phase-5 supervised `max_descent_mm=None` with operator `Ctrl+C` as the absolute
  depth guard is not sufficient for a formal autonomous Phase-8 runtime. The first
  E2E pilot reached `Z=-52 commanded-mm` before SIDE latched; physical success does
  not remove the need for a meaningful software/workspace depth safety fuse.
- The `g/Q` OCR confusion was not actually solved by earlier Phase-5 voting. Real
  saved released-character crops must become regression fixtures, and any OCR fix
  must remain target-independent rather than hard-coding `G`.

### Next closing work

Tomorrow's closing sequence should remain narrow and evidence-driven:

1. remove validation-only fixed-prefix and normal-path human arming from the formal
   Phase-8 runtime while keeping `Ctrl+C` / emergency recovery behavior;
2. define and test a physically meaningful autonomous absolute-depth/workspace
   bound without changing the fixed Goal-anchor semantics;
3. turn the saved `released_char_*.png` / `event_novel_crop.png` from the first E2E
   run into offline OCR regression fixtures and fix the `g/Q` failure without
   expected-target bias;
4. rerun `G` end to end and require both physical correctness **and** software
   `SUCCESS`, followed by HOME PASS;
5. only then run a small cross-key Phase-8 transfer set (start with already accepted
   deterministic keys such as `F` / `H`) without per-key tuning;
6. freeze Phase-7/8 acceptance evidence, tests, README, and repository history before
   beginning Phase 9 multi-key typing.

<!-- IMPLEMENTATION_PROGRESS:END -->


------------------------------------------------------------------------

# Development Roadmap

The roadmap is updated to reflect the current physical progress.

## Phase 0 — Mechanical Feasibility — **Completed**

Goal:

> Verify that the existing SO-101/tool setup can physically press
> MacBook keys.

Already demonstrated through teleoperation.

No redesign of the end effector is required before software work begins.

------------------------------------------------------------------------

## Phase 1 — Freeze Camera Geometry and Build Camera Sanity Tool

Goal:

> Validate the final physical camera layout and all preprocessing
> assumptions.

Implement a tool that displays:

``` text
TOP raw
TOP optional screen mask / leakage statistics

WRIST raw
WRIST candidate keycap overlays

SIDE raw
SIDE rectified screen
SIDE text ROI
```

Acceptance:

- all three cameras stable at 30 FPS under the intended configuration,
- TOP covers useful robot/keyboard workspace,
- WRIST resolves local keycaps/glyphs,
- SIDE rectification produces readable screen text,
- frame timestamps are available,
- camera mounts are frozen after acceptance.

------------------------------------------------------------------------

## Phase 2 — Wrist Keycap Detection and Glyph Recognition

Goal:

> Given a wrist frame and a requested character, visually locate that
> character without using a pre-programmed keyboard-position identity.

Implement:

``` text
keycap segmentation
      ↓
candidate extraction
      ↓
per-key rectification
      ↓
glyph recognition
      ↓
TargetObservation
```

Compare:

``` text
Template Matching
HOG + SVM
Tiny CNN
```

Acceptance should measure:

- keycap detection precision/recall,
- character classification accuracy,
- target acquisition success rate,
- false acquisition rate,
- confidence calibration,
- performance under small pose/exposure variations.

------------------------------------------------------------------------

## Phase 3 — Tool Reference and Visual Servo — **Completed**

Goal:

> Starting from a safe teleoperated local pose, align the detected target key
> to a directly calibrated WRIST pencil-tip reference and validate a fixed-anchor,
> dead-zone-aware staged XYZ primitive.

Accepted scope:

1.  direct physical WRIST `p_tip` calibration,
2.  canonical H3.2 command-space `J_cmd`,
3.  motion-stable handoff that preserves existing `Goal_Position` as the fixed
    command-space anchor,
4.  bounded cumulative `p_key -> p_tip` XY correction with fresh-frame control,
5.  semantic target lock plus guarded geometry tracking through near-target glyph
    occlusion,
6.  stable multi-frame XY acceptance,
7.  cumulative staged Z levels from the same Goal anchor,
8.  stop/settle/fresh-WRIST observation after every Z-level change,
9.  same-level XY re-alignment without relatching the command origin.

Hardware acceptance evidence includes:

- direct `p_tip ~= (307.0, 238.246) px`,
- demonstrated XY capture from approximately 61.9 px residual to 4.56 px,
- a separate stable 1.79 px / 2-frame XY acceptance,
- cumulative Z `0 -> -3 -> -6 commanded-mm`,
- successful low-Z XY re-alignment,
- final staged-Z visual error of 2.88 px,
- `Z stage status = complete`.

Planning adjustment after hardware evidence:

- `Present_Position`-based runtime rebasing is rejected; existing
  `Goal_Position` is the deterministic command anchor.
- XYZ are all treated as dead-zone/backlash affected; cumulative commands are
  preserved across steps.
- Cross-key end-to-end transfer is no longer a Phase 3 acceptance gate. It is
  more meaningfully exercised in Phase 5 once screen verification can confirm
  real physical outcomes on multiple keys.

No ACT and no screen-confirmed keypress success are required for Phase 3.

------------------------------------------------------------------------

## Phase 4 — Screen Rectification and Verification — **Completed**

Goal:

> Reliably determine whether a stopped physical press level produced the
> expected screen change.

Accepted pipeline:

``` text
SIDE
  ↓
screen homography
  ↓
canonical screen
  ↓
fixed typing ROI
  ↓
dark text-line detection
  ↓
3x line crop + CLAHE
  ↓
Tesseract 5.3.4 / OEM 1 / PSM 7 / eng / A-Z0-9 whitelist
  ↓
confirmed-prefix + expected-character semantic comparison
  ↓
9-frame vote (60% confirmation threshold)
  ↓
CONFIRMED_SUCCESS / CONFIRMED_NO_CHANGE /
CONFIRMED_WRONG / UNCERTAIN
```

Phase 1 assets were reused unchanged:

- fixed SIDE camera geometry,
- `calibration/screen_homography.json`,
- canonical `1280x800` rectified screen,
- fixed `960x694` typing ROI.

Phase 4 deliberately rejected whole-ROI exact-string OCR consensus after live
experiments showed only `2/9` exact matches despite readable content. The accepted
scheme localizes real text lines first and uses Tesseract only as an observation
engine; the verifier then compares against the already-confirmed prefix and the
expected next character. It never repairs OCR output into the expected answer.

Live acceptance evidence:

- `KEYPRESS` -> `CONFIRMED_NO_CHANGE`, **9/9** votes;
- `KEYPRESSG` -> `CONFIRMED_SUCCESS`, **9/9** votes;
- `KEYPRESSH` -> `CONFIRMED_WRONG`, **9/9** votes with wrong character `H`;
- `UNCERTAIN` behavior for disagreement / insufficient evidence is covered by
  unit tests.

This completes the independent screen-verification authority required by the
Phase 5 physical press loop.

------------------------------------------------------------------------

## Phase 5 — Deterministic Local Single-Key Closed Loop — **Completed — G/F/H cross-key acceptance**

Goal:

> From a safe local pose near a requested key, visually acquire and align the
> target, descend in small same-anchor command increments, detect the first
> screen-change evidence with independent SIDE vision, release immediately,
> verify the released character, and retract safely.

Current implemented loop:

``` text
target=G
   ↓
WRIST recognizes G and latches semantic identity
   ↓
p_key -> p_tip visual alignment
   ↓
iterative cumulative Z step = -2 commanded-mm
   ↓
stop + fresh WRIST / same-Z XY realignment as needed
   ↓
SIDE watcher continuously checks continuation-region change
   ├── no event + stable NO_CHANGE → one more Z step
   ├── strong new glyph → latch on first changed frame
   └── ordinary smaller change → latch after 2 consecutive frames
           ↓
       PRESS_EVENT_LATCHED
           ↓
       permanently forbid more Z-down / XY chase
           ↓
       use latest actually-sent XYZ command state
           ↓
       one +20 commanded-mm upward release
           ↓
       single-character OCR on released new component
           ↓
       SUCCESS / WRONG / UNCERTAIN
           ↓
       segmented retract (<=10 commanded-mm Z per segment)
```

### Current hardware checkpoint

The current G run passed cleanly:

- strong SIDE event detected on the first changed frame;
- event evidence was approximately `765` novel pixels with a largest connected
  component of approximately `765 px`;
- event-to-release-command latency was approximately `17.6 ms`;
- release preserved the latest actually-sent XY command state;
- release changed command-space Z from `-42` to `-22` in one upward escape;
- released-character OCR produced `G / Q / G / G / Q`, resulting in
  `CONFIRMED_SUCCESS`;
- this vote should **not** be interpreted as a solved single-character OCR problem:
  the later Phase-8 E2E pilot produced `Q / Q / Q / Q / Q` for a visibly correct
  lowercase `g`, proving that the old vote only masked the remaining `g/Q` weakness;
- segmented retract returned command-space Z to `0`;
- final controller state was `SUCCEEDED`, outcome `SUCCESS`.

The earlier event-interrupt stale-state bug is fixed. The release path no longer
uses the stale outer control state when an event arrives during an inner WRIST
motion; it uses the latest command that was actually sent to the robot.

### Low-level tracking note

Phase 5 does not require millimetre-level physical TCP tracking from the SO-101.
As already established in Phase 3, command-space millimetres are control units,
not a guarantee of equal measured TCP displacement.

Servo telemetry from the latest G checkpoint showed that the upward release
produced enough physical motion to cross the keyboard release point, while the
arm became motion-stable before `Present_Position` fully converged to
`Goal_Position`. Load-dependent Goal/Present tracking, backlash/compliance,
motion-completion semantics, and possible contact/load sensing are therefore
deferred performance optimizations, not blockers for the deterministic typing
architecture.

### Cross-key acceptance

Phase 5 acceptance is complete. The same deterministic loop passed on `G`, `F`,
and `H` without per-key tuning of `p_tip`, `J_cmd`, OCR, tracking, SIDE-event, or
press/release parameters. This freezes the deterministic local controller as the
downstream takeover target for Phase 6 replay validation.

Keep `scripts/validate_phase3_xyz.py` as the Phase 3 hardware regression validator
and `scripts/validate_phase5_single_key.py` as the Phase 5 hardware
acceptance/regression runner.

Current supervised validation intentionally has no arbitrary cumulative-Z
software cap (`max_descent_mm=None`); the operator remains the absolute-depth
safety authority. A future autonomous version must add a physically meaningful
safety bound without requiring ACT to start at a fixed distance from the
keyboard.

### Phase 5 code organization

The following modules are intended implementation code:

- `fixed_anchor_xyz.py`
- `fixed_anchor_planner.py`
- `semantic_target_lock.py`
- `screen_change.py`
- `press_controller.py`
- `validate_phase5_single_key.py`
- their focused unit/regression tests.

Temporary patch installers, generated telemetry/debug outputs, and backup
directories under `artifacts/` are transition/debug artifacts and should not be
committed.

## Phase 6 — Target-Conditioned ACT Dataset — **Completed**

Goal:

> Build a target-conditioned ACT dataset whose demonstrations are natural human
> coarse-approach trajectories **and whose episode endpoints are physically
> verified to support deterministic WRIST-servo takeover**.

Phase 6 is complete as an episode-level collection, qualification, and physical
validation pipeline.

### 6A — Generic natural episode collection

The retained collector is intentionally neutral with respect to collection
strategy. It does not know or enforce semantic start classes such as HOME, LEFT,
RIGHT, FRONT, or RETRACT-LIKE. Coverage of the initial-state distribution belongs
to an external experiment SOP or future scheduler.

The collector owns only generic episode mechanics:

- target-conditioned episode recording;
- automatic arm/countdown timing;
- true-motion start detection after arming;
- TOP + WRIST + robot state + target condition capture;
- actual `robot.send_action(...)` result stored as the action;
- operator-controlled natural episode end with `s`/SPACE;
- discard/retry support with `q`;
- preparation/reaction tail exclusion from the saved demonstration;
- 15 Hz ACT samples plus a separate approximately 60 Hz actual-sent-action trace
  for physical replay;
- audit/performance diagnostics stored separately from policy observations.

No SIDE pixels, press actions, OCR results, proximity cue, fixed `d_tip` endpoint,
or autonomous motion belong in the formal ACT demonstration stream.

### 6B — Automatic offline episode QC

Offline QC evaluates the episode itself, not the collection strategy that
produced its start state. Example checks include:

``` text
schema / image / state / action completeness
valid target encoding
actual sent-action continuity
unintended long pauses / hesitation
strong repeated reversals
abnormal duration
significant requested-to-sent clamp
control-loop timing / sample misses
```

A QC PASS produces a replay candidate. REVIEW/FAIL episodes remain in the raw
source dataset for audit/debugging but are excluded from the verified training
manifest unless later replaced or explicitly requalified.

### 6C — Full-rate physical replay and deterministic takeover

Each QC-PASS candidate is validated on hardware:

``` text
restore recorded episode start
        ↓
replay the recorded actual sent-action sequence with original timing
        ↓
reach the recorded episode endpoint with its command history
        ↓
transfer ownership directly to deterministic WRIST servo
        ↓
run real local XY takeover
        ├── FAIL    → exclude episode
        └── SUCCESS → verified episode
```

The validator does not teleport to the final joint vector. Full replay is needed
because SO-101 dead zone, backlash, compliance, preload, and Goal/Present lag make
endpoint behavior history-dependent.

Phase 6 takeover acceptance deliberately stops at **stable WRIST XY alignment**.
It does not descend Z, press the key, run SIDE event detection, release, OCR, or
retract. Those mechanics are already validated in Phase 5 and are integrated
with the learned policy later in Phase 8.

For deterministic takeover, the existing servo `Goal_Position` remains the fixed
command-space anchor. `Present_Position` may be used before an experiment for
safe transport initialization and diagnostics, but is not repeatedly promoted to
a new precision-control origin.

### 6D — Batch validation and recovery contract

Physical validation is batch-oriented:

``` text
explicit recovery HOME
        ↓
episode A start -> replay -> WRIST takeover
        ↓
current physical pose -> episode B start
        ↓
episode B replay -> WRIST takeover
        ↓
...
        ↓
final explicit recovery HOME
        ↓
safe disconnect
```

There is no HOME reset between candidate episodes. The final recovery pose is
robot configuration, not dataset semantics, and is stored explicitly at:

``` text
configs/robot/recovery_home.json
```

`scripts/export_episode_start_pose.py` is a generic pose-export utility used to
create such robot pose configuration from a deliberately selected known-safe
recorded pose. The replay validator itself never searches for a dataset episode
whose label happens to mean HOME.

### 6E — Final verified dataset contract

Initial policy feature contract:

``` text
observation.images.top    : RGB
observation.images.wrist  : RGB
observation.state         : 6 Present joint positions + 28D target one-hot
action                    : 6 actual sent absolute joint-position goals
task                      : approach_key:<TARGET>
```

Initial target vocabulary:

``` text
A-Z, SPACE, BACKSPACE
```

The SIDE/SCREEN camera is excluded from ACT observations.

The final acceptance unit is the whole episode:

``` text
natural human demo
+ offline QC PASS
+ full physical replay
+ deterministic WRIST-servo takeover PASS
= verified ACT episode
```

### Phase 6 acceptance evidence

The Phase 6 G dataset now contains five physically verified episodes:

``` text
dataset episodes: 0, 1, 2, 3, 5
```

Raw episode `4` was retained but excluded after offline QC reported a possible
hesitation pause. The replacement episode `5` provided a clean final acceptance
case:

- collection: 49 saved 15 Hz samples over approximately `3.21 s`;
- collection control: approximately `59.8 Hz`, no loop overruns, no sample misses,
  no requested-to-sent action difference;
- offline QC: PASS;
- physical replay: 192 full-rate control samples, `0.000 deg` maximum sent-action
  difference, approximately `2.0 ms` maximum schedule lateness;
- WRIST takeover: approximately `27.29 px` initial residual;
- stable takeover acceptance: approximately `2.91 px` then `2.77 px`, satisfying
  the required `2/2` fresh-frame tolerance condition;
- final batch recovery: explicit recovery HOME restored with approximately
  `0.044 deg` Goal difference before disconnect.

Earlier verified episodes exercised substantially different operator-selected
initial poses and also completed full physical replay followed by deterministic
WRIST convergence. Those initial-state categories were part of the experiment
strategy only and are not program semantics.

Phase 6 is therefore frozen as **Completed**. Phase 7 consumes only the verified
episode manifest and begins ACT training/inference work; the rejected/review raw
episodes remain available for debugging but must not silently enter training.

------------------------------------------------------------------------

## Phase 7 — ACT Coarse Policy

Goal:

> Given TOP + WRIST + robot state + requested key, move to a servo-ready
> local viewpoint using one shared target-conditioned ACT policy.

ACT does not need to press the key. Deterministic WRIST visual servoing remains
responsible for final local XY alignment, and the Phase 5 press/SIDE/OCR/retract
loop remains outside ACT.

### 7A — Verified-only training input — **Completed**

The Phase 6 source dataset is immutable. A verified manifest selects eligible
training episodes, and Phase 7 computes normalization statistics from exactly that
same subset rather than from raw repository-wide metadata.

Current formal contract:

``` text
source dataset         = artifacts/phase6_act_dataset/keyboard_v1
verified manifest      = phase6.verified_episode_manifest.v4
verified episodes      = dynamic from manifest
verified frames        = dynamic from selected episodes
LeRobot                = 0.6.1
```

Retained implementation:

``` text
scripts/phase7_act_training_input.py
tests/test_phase7_act_training_input.py
```

### 7B — ACT temporal contract — **Completed**

The training-side temporal horizon is frozen at:

``` text
fps                         = 15 Hz
chunk_size                  = 20
first-to-last action span   = 1.267 s
20-command duration         = 1.333 s
episode-tail padding        = recomputed from formal verified subset
full-window coverage        = recomputed from formal verified subset
```

`n_action_steps` is a separate runtime replanning/execution-horizon decision and
is intentionally not frozen during dataset construction/training-input work.

Retained implementation:

``` text
scripts/phase7_act_temporal_contract.py
tests/test_phase7_act_temporal_contract.py
```

### 7C — ACT training-path integration — **Completed**

The project uses official LeRobot ACT internals rather than a reimplementation of
ACT. `scripts/phase7_train_act.py` supplies the project-specific verified subset,
verified-only normalization, and frozen temporal contract, then calls the official
LeRobot dataset/model/processor/optimizer/checkpoint APIs.

Validated configuration includes:

``` text
ACT backbone             = ResNet18
backbone initialization  = ResNet18_Weights.IMAGENET1K_V1
batch_size               = 8
chunk_size               = 20
verified normalization   = exact formal verified subset only
```

A real one-step forward/backward/optimizer/checkpoint run passed on the RTX 5070
Ti with approximately `3.19 GiB` peak CUDA allocation. The earlier standalone
forward-only smoke script is no longer retained because the training entry point
now subsumes that test.

### 7D — Formal target-conditioned keyboard dataset — **Completed**

This subsection preserves the formal V1 dataset/coverage contract that produced the
retained `keyboard_v1` training line. The final verified manifest and matching
verified-only normalization provenance used by the retained 20k checkpoint are
frozen in the Phase-7 acceptance evidence below.

The V1 dataset is designed as one shared target-conditioned policy dataset with a
28D target schema but an A-Z-only first collection pass:

``` text
schema targets = A-Z + SPACE + BACKSPACE (28D reserved vocabulary)
V1 collected targets = A-Z (26 targets)
6 representative source zones per collected target
1 verified episode per source-target pair
---------------------------------------------------
156 verified base episodes
```

`SPACE` and `BACKSPACE` are not part of the V1 quota. They can be appended later
to the same schema/dataset family after deterministic WRIST takeover support is
validated for those keys.

Representative source zones:

``` text
S0  HOME / safe startup
S1  upper-left keyboard region     (~Q/W/E)
S2  upper-right keyboard region    (~I/O/P/BACKSPACE)
S3  center keyboard region         (~F/G/H/J)
S4  lower-left keyboard region     (~Z/X/C)
S5  lower-right keyboard region    (~B/N/M/SPACE)
```

The source zones are external SOP regions only. They are not labels in the ACT
dataset and are not program semantics in the collector/QC/replay tools. Each zone
allows natural position, height, and joint-configuration variation. Height is not
a separate combinatorial quota.

Each episode remains subject to the existing Phase 6 acceptance funnel:

``` text
natural demonstration
        ↓
offline QC PASS
        ↓
full physical replay of actual sent actions
        ↓
real deterministic WRIST takeover PASS
        ↓
verified formal training episode
```

After the frozen verified base manifest is trained and evaluated, additional data
should be collected only for source-target regions that show actual rollout failures,
rather than uniformly expanding every planned source-target cell.

### Trained-policy runtime rollout checkpoint

The first formal runtime checkpoint now exists at:

``` text
artifacts/phase7_act_coarse/keyboard_v1/act_train_20k/checkpoints/last
```

`scripts/phase7_act_rollout.py` is the retained trained-policy hardware evaluator.
Its current contract is intentionally diagnostic rather than a Phase-8 state
machine:

- official LeRobot `ACTPolicy.select_action()` queue;
- 15 Hz control, `chunk_size=20`;
- current baseline `n_action_steps=20`;
- moving target `<=80 px` produces `HANDOFF_CANDIDATE`, not `SERVO_READY`;
- ACT is stopped/reset before deterministic ownership;
- a fresh post-settle `SETTLED_ENDPOINT` is recorded;
- the existing Phase-6 deterministic WRIST takeover remains the ground-truth
  acceptance test;
- the evaluator does **not** press a key and does not run SIDE/OCR.

Final hardware evidence spans `G`, `F`, `Q`, `P`, and `M` with the same retained
checkpoint and `n_action_steps=20`. The important result is not that `80 px` has
been proven as a universal capture threshold; it has not. The moving threshold
remains only a handoff-candidate trigger, while actual deterministic WRIST
convergence remains the acceptance ground truth.

The two retained `Z` timeouts are documented as a target-perception limitation:
the physical ACT approach reached the Z neighborhood, but the runtime WRIST glyph
recognizer produced no accepted `Z` observation. They are not counted as PASS and
are not used to justify target-specific ACT tuning.

### Phase 7 acceptance closeout

Phase 7 is accepted.

The final provenance chain is frozen:

- `phase6.verified_episode_manifest.v4`;
- 158 verified episodes / 7561 verified frames;
- verified-only normalization from exactly the same selected episode subset;
- official LeRobot ACT training path with ImageNet-pretrained ResNet18;
- 20,000 completed optimization steps;
- retained `checkpoints/last -> step_020000`;
- offline 26-target diagnostic validation PASS;
- runtime latency and observation-to-action age characterized from hardware
  rollout evidence;
- retained runtime acceptance baseline: 15 Hz, `chunk_size=20`,
  `n_action_steps=20`, Temporal Ensemble OFF.

Final retained hardware rollout evidence:

| Target/run | Moving candidate | Settled endpoint | Result |
|---|---:|---:|---|
| `G` v6 run 1 | `78.47 px` | `76.92 px` | PASS |
| `G` v6 run 2 | `78.34 px` | `72.40 px` | PASS |
| `F` v6 | `65.68 px` | `59.28 px` | PASS |
| `Q` v6 | `40.60 px` | `38.78 px` | PASS |
| `P` v6 | `75.16 px` | `72.25 px` | PASS |
| `M` v6 | `73.64 px` | `63.19 px` | PASS |

Every PASS above used the same retained checkpoint and runtime configuration,
reached real deterministic WRIST convergence, and completed automatic HOME
recovery.

Two `Z` runs were also retained. Both ended in `TIMEOUT_NO_TAKEOVER`; they are
**not** recorded as PASS. Saved WRIST evidence shows that the physical ACT approach
reached the Z neighborhood, but the runtime WRIST glyph recognizer never produced
an accepted `Z` observation. This is frozen as a known target-perception limitation
for later work. No Z-specific ACT parameter tuning, safety-limit relaxation, or
blind `n_action_steps` search is introduced to make the case pass.

Phase 7 therefore closes on demonstrated multi-target ACT coarse approach and
successful deterministic-controller handoff. Z descent, physical keypress,
SIDE-event handling, released-character OCR, and final key outcome remain
Phase-8 acceptance concerns.

------------------------------------------------------------------------

## Phase 8 — ACT + Visual Servo Handoff — **In Progress**

Goal:

> Integrate the learned and deterministic controllers without command overlap,
> then preserve one deterministic fixed-anchor primitive through physical press,
> independent screen evidence, release, semantic verification, retract, and HOME.

Formal pipeline:

``` text
ACT_APPROACH
     ↓
target perception
     ↓
HANDOFF_CANDIDATE
     ↓
stop + settle + fresh endpoint diagnostic
     ↓
flush/reset ACT; deterministic ownership only
     ↓
latch existing Goal_Position once
     ↓
SERVO_ALIGN
     ↓
small cumulative DESCEND_STEP
     ↓
STOP / REOBSERVE / SAME-Z REALIGN
     ↓
SIDE watcher
     ├── no event + confirmed NO_CHANGE → next descent step
     └── persistent screen change → EVENT LATCH → RELEASE FIRST
                                      ↓
                                released-char verify
                                      ↓
                                retract / supervisor
                                      ↓
                                automatic HOME
```

### First full-integration evidence

The first `G` pilot has already exercised the complete physical path. It should be
recorded as **physical-loop success but software acceptance failure**, not as a
completed Phase-8 PASS:

- ACT coarse approach ran from the trained 20k policy with `n_action_steps=20`;
- deterministic ownership began after ACT stop/reset;
- WRIST/Z/release/retract stayed on one fixed existing Goal anchor;
- the high-priority SIDE event fired and the release command was sent about
  `18.3 ms` after event capture;
- the event interrupted an inner WRIST path and the release correctly used the
  latest actually-sent XYZ state instead of stale outer state;
- the physical editor result observed by the operator was the requested lowercase
  `g`;
- single-character OCR nevertheless returned `Q` on `5/5` released frames;
- controller outcome was therefore `WRONG_KEY` / `FAIL_PRESS_OUTCOME`;
- retract completed and HOME recovery passed with about `0.044 deg` final Goal
  difference.

The run also exposed a safety/formalization issue: the event arrived at cumulative
command-space `Z=-52 mm`. Commanded millimetres are not physical TCP millimetres,
but the formal autonomous runtime still requires a meaningful absolute-depth or
workspace safety fuse. Operator `Ctrl+C` remains an emergency stop, not the desired
production safety policy.

### Formalization requirements before Phase-8 acceptance

- no fixed `KEYPRESS` or other validation-only prefix in normal runtime logic;
- no normal-path `Press ENTER` / `Type GO` operator gate between ACT and the
  deterministic primitive;
- `Ctrl+C` / emergency hold and recovery safety behavior remain available;
- support or explicitly constrain the empty-screen baseline case instead of
  silently assuming detectable pre-existing text;
- offline regression-test released-character OCR using real saved crops from
  hardware runs before repeating expensive physical trials;
- do not bias OCR toward the expected target; WRONG/UNCERTAIN must remain real
  outcomes;
- define the autonomous Z/workspace safety fuse without relatching
  `Present_Position` or changing the fixed Goal anchor;
- rerun `G` until the software verdict agrees with the physical result;
- then confirm transfer on multiple keys without per-key tuning.

Acceptance must explicitly test:

- no stale ACT commands after handoff,
- deterministic controller ownership,
- one fixed Goal anchor across align/Z/release/retract,
- target reacquisition behavior,
- SIDE-event-to-release latency,
- autonomous depth/workspace safety behavior,
- released-character OCR accuracy / false-WRONG rate,
- handoff success rate,
- end-to-end single-key software success rate,
- HOME/recovery success on both success and failure outcomes.

------------------------------------------------------------------------

## Phase 9 — Multi-Key Typing

Goal:

> Repeatedly execute the single-key primitive for short strings.

Initial examples:

``` text
CAT
DOG
HELLO
ROBOT
VISION
```

Verify after every character.

------------------------------------------------------------------------

## Phase 10 — Automatic Recovery

Goal:

> Detect incorrect physical outcomes and repair them automatically.

Evaluate:

- wrong-key detection,
- uncertain OCR handling,
- recovery success rate,
- average corrective actions,
- final corrected-string accuracy.

------------------------------------------------------------------------

## Phase 11 — Controlled Generalization and Ablations

Only after the fixed setup is reliable.

Candidate experiments:

- robot initial configuration variation,
- small camera perturbations,
- lighting variation,
- keyboard translation,
- keyboard rotation,
- unseen target strings,
- TOP raw vs TOP screen-masked,
- ACT only vs visual servo only vs ACT + visual servo,
- template vs HOG/SVM vs tiny CNN,
- different ACT handoff thresholds,
- different servo gains and step bounds.

------------------------------------------------------------------------

# Evaluation

## End-to-End Typing

Measure:

``` text
single-key success rate
character success rate before recovery
character success rate after recovery
word success rate
final task success rate
characters per minute
average retries per character
```

## Wrist Perception

Measure:

``` text
keycap detection precision / recall
glyph classification accuracy
target acquisition success rate
false target acquisition rate
confidence
target pixel size at acquisition
occlusion rate
```

## ACT

Measure:

``` text
raw episode collection acceptance rate
offline-QC pass rate
full-replay success rate
WRIST-servo takeover success rate
verified-episode yield
coarse approach success rate after training
target visibility after ACT
approach time
failure / timeout rate
```

## Visual Servo

Measure:

``` text
initial pixel error
final pixel error
servo iterations
convergence time
convergence rate
target-loss rate
```

## Verification and Recovery

Measure:

``` text
screen OCR accuracy
false SUCCESS rate
false WRONG rate
UNCERTAIN rate
recovery success rate
corrective action count
final corrected-string accuracy
```

------------------------------------------------------------------------

# Safety

The robot operates directly above a laptop, so all motion near the
keyboard must be bounded.

Required safeguards:

``` text
joint limits
workspace limits
maximum Cartesian correction per servo iteration
maximum cumulative Z-level increment
production/autonomous absolute-depth or workspace safety bound
maximum press duration
maximum servo iterations
fresh-frame requirement after every Z step
frame-age threshold
target-confidence threshold
controller-ownership lock
timeout conditions
emergency stop
```

Safety invariants:

1.  no valid target → no downward step,
2.  target lost → no downward step,
3.  stale image → no servo correction and no downward step,
4.  alignment not stable at the current stopped level → no downward step,
5.  a downward level transition changes cumulative Z while holding cumulative XY
    fixed; any lateral correction occurs only after stop/settle at that fixed Z,
6.  deterministic commands remain cumulative from one fixed Goal-space anchor;
    `Present_Position` is never silently promoted to a new runtime command origin,
7.  ACT and the deterministic staged controller never command simultaneously,
8.  a latched persistent SIDE press event permanently disables further Z-down and
    XY chase for that attempt,
9.  release preserves the latest authoritative cumulative XY and changes only Z,
10. failed/uncertain verification never causes an unbounded descent or retry loop.

------------------------------------------------------------------------

# Current Software Environment

The current validated development environment is approximately:

``` text
Python       3.12.14
LeRobot      0.6.1
PyTorch      2.11.0 + CUDA 13.0 build
TorchVision  0.26.0 + CUDA 13.0 build
TorchCodec   0.11.1
Feetech SDK  1.0.0
OpenCV       4.13.0 (headless)
GPU          NVIDIA GeForce RTX 5070 Ti
VRAM         ~15.4 GiB
```

## For reproducibility, the project should pin the exact LeRobot version/commit and keep an exported environment file.

# Proposed Repository Structure

``` text
so101_typing/
│
├── configs/
│   ├── cameras/
│   │   ├── top.yaml
│   │   ├── wrist.yaml
│   │   └── screen.yaml
│   ├── robot/
│   │   └── recovery_home.json
│   ├── perception/
│   ├── visual_servo/
│   ├── press/
│   └── act/
│
├── calibration/
│   ├── screen_homography.json
│   ├── tool_reference.json
│   ├── image_jacobian.json
│   └── top_screen_mask.json
│
├── src/
│   └── so101_typing/
│       ├── adapters/
│       │   ├── robot.py
│       │   └── cameras.py
│       │
│       ├── perception/
│       │   ├── keycaps.py
│       │   ├── glyphs.py
│       │   ├── target_observation.py
│       │   ├── semantic_target_lock.py
│       │   ├── screen_rectify.py
│       │   ├── screen_ocr.py
│       │   └── screen_change.py
│       │
│       ├── control/
│       │   ├── visual_servo.py
│       │   ├── image_jacobian.py
│       │   ├── fixed_anchor_xyz.py
│       │   ├── fixed_anchor_planner.py
│       │   ├── controller_owner.py
│       │   └── safety.py
│       │
│       ├── policy/
│       │   ├── act_policy.py
│       │   ├── target_encoding.py
│       │   └── dataset.py
│       │
│       ├── supervisor/
│       │   ├── state_machine.py
│       │   ├── typing.py
│       │   ├── press_controller.py
│       │   ├── verification.py
│       │   └── recovery.py
│       │
│       ├── runtime/
│       │   ├── single_key.py
│       │   ├── typing.py
│       │   ├── event_log.py
│       │   └── evaluate.py
│       │
│       └── utils/
│           ├── timing.py
│           └── visualization.py
│
├── scripts/
│   ├── camera_sanity.py
│   ├── calibrate_screen.py
│   ├── calibrate_tool_reference.py
│   ├── calibrate_image_jacobian.py
│   ├── collect_wrist_dataset.py
│   ├── benchmark_glyph_models.py
│   ├── validate_phase3_xyz.py
│   ├── validate_phase4_screen_verification.py
│   ├── validate_phase5_single_key.py
│   ├── phase6_act_dataset_collector.py
│   ├── phase6_episode_qc.py
│   ├── phase6_replay_takeover_validate.py
│   ├── export_episode_start_pose.py
│   ├── phase7_act_training_input.py
│   ├── phase7_act_temporal_contract.py
│   ├── phase7_train_act.py
│   ├── phase7_act_rollout.py
│   ├── phase8_single_key_integration.py
│   └── run_typing_demo.py
│
├── tests/
│   ├── test_screen_ocr.py
│   ├── test_screen_line_ocr.py
│   ├── test_screen_verification.py
│   ├── test_state_machine.py
│   ├── test_recovery.py
│   ├── test_target_encoding.py
│   ├── test_controller_ownership.py
│   ├── test_phase7_act_training_input.py
│   └── test_phase7_act_temporal_contract.py
│
├── docs/
│   ├── architecture.md
│   ├── calibration.md
│   ├── perception.md
│   ├── dataset.md
│   └── experiments.md
│
└── README.md
```

------------------------------------------------------------------------

# First End-to-End Milestone

The primary learned-to-deterministic architecture has now been exercised on real
hardware as one continuous single-key run. The first `G` Phase-8 pilot physically
completed the complete chain:

``` text
target G
   ↓
trained ACT coarse approach
   ↓
ACT stop/reset + deterministic ownership
   ↓
fixed Goal-space anchor
   ↓
WRIST visual servo / geometry fallback
   ↓
iterative Z descent + same-level XY re-alignment
   ↓
SIDE persistent screen-change event
   ↓
release command ~18.3 ms after event capture
   ↓
physical editor receives g
   ↓
segmented retract to command-space Z=0
   ↓
automatic HOME PASS
```

This is the first evidence that the **physical hybrid loop itself** works from the
learned policy all the way through contact and recovery.

It is not yet the complete software milestone. The same run ended with:

``` text
released-character OCR = Q / Q / Q / Q / Q
expected                = G
controller outcome      = WRONG_KEY
Phase-8 task status     = FAIL_PRESS_OUTCOME
```

The end-to-end milestone will be considered complete only when physical outcome,
independent SIDE semantics, software verdict, retract, and HOME all agree on the
same run without validation-only manual gates or fixed text fixtures.

Remaining closing work before that milestone:

``` text
formalize Phase-8 runtime (no fixed KEYPRESS / no normal GO prompt)
        ↓
add autonomous absolute-depth/workspace safety fuse
        ↓
fix released-char g/Q OCR from real saved regression crops
        ↓
rerun G: physical G + software SUCCESS + retract + HOME
        ↓
repeat a small cross-key acceptance set without per-key tuning
        ↓
freeze Phase-7/8 metrics, tests, README, and repository checkpoint
```

After that, Phase 9 multi-character typing becomes repeated execution of a proven
single-key primitive plus supervisor-driven recovery rather than a new control
architecture problem.

# Final Demonstration

A target such as:

``` text
TARGET: ROBOT
```

is provided to the supervisor.

The system repeatedly executes:

``` text
target character
      ↓
ACT coarse approach
      ↓
appearance-based wrist recognition
      ↓
perception-triggered handoff
      ↓
settle + preserve existing Goal_Position command anchor
      ↓
p_key -> p_tip visual alignment
      ↓
fixed-anchor cumulative Z / stop / reobserve / same-level realign
      ↓
SIDE fast event latch on first persistent screen change
      ↓
release escape + released-character verification
      ↓
segmented retract / recovery
      ↓
next character / recovery
```

If the robot produces:

``` text
ROBOR
```

the external screen observer detects the mismatch and the supervisor
generates:

``` text
BACKSPACE
T
```

until the display contains:

``` text
ROBOT
```

The final objective is therefore not simply:

> make a robot press keyboard keys

but:

> **build a robot that perceives a target, uses learning to create a
> useful local state, switches to explicit closed-loop control for
> precision, physically acts on the world, independently observes the
> consequence, and autonomously corrects errors.**

------------------------------------------------------------------------

# Long-Term Extensions

After the V0 architecture is reliable:

- SmolVLA or another language-conditioned policy,
- arbitrary target phrases,
- shifted keyboard poses,
- multiple keyboard models,
- punctuation and modifier keys,
- learned key detectors,
- uncertainty-aware perception,
- continuous ACT + visual-servo residual control,
- learned recovery,
- multi-camera policy fusion,
- online replanning,
- autonomous calibration,
- transfer to other button-based physical interaction tasks.

------------------------------------------------------------------------

# Project Principle

``` text
string decomposition      → deterministic supervisor
target selection          → deterministic supervisor
coarse robot motion       → ACT
local key identity        → visual appearance
fine alignment            → classical visual servo
physical press            → bounded staged deterministic control
result observation        → SIDE fast event + independent screen OCR
recovery planning         → deterministic supervisor
```

The project is intentionally hybrid.
Learning is used where variation and motion priors are valuable.

Classical algorithms are used where geometry, feedback, safety, and task
semantics can be made explicit.

The boundary between them is not hidden: it is part of the system being
studied.
