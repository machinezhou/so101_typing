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

Phases 0–4 are accepted. Phase 5 remains the current implementation phase and
its deterministic single-key `G` loop now has a clean hardware SUCCESS checkpoint
with the high-priority SIDE-event release path operating as intended. Phase 5 is
not marked complete yet because cross-key transfer on at least two additional
letters is still required.

The current physical/software setup has demonstrated:

- calibrated SO-101 follower/leader operation and stable teleoperation,
- a previous ACT block-manipulation demo,
- a pencil tool mechanically attached to the gripper,
- three simultaneous camera streams in the fixed MacBook workspace,
- appearance-based WRIST target acquisition,
- fixed-`Goal_Position` cumulative Cartesian command semantics,
- WRIST visual servoing with guarded semantic/geometry tracking through glyph
  occlusion,
- independent SIDE screen rectification and conservative semantic verification,
- a high-priority SIDE watcher with two event paths:
  - a strong continuation-region glyph can latch on the first frame,
  - ordinary smaller changes still require two consecutive frames,
- immutable event ownership once latched: no later observation can authorize
  additional Z-down or XY chase for that press attempt,
- release from the latest actually-sent XYZ command state, eliminating the
  stale-outer-state XY rollback found in the earlier hardware run,
- one upward `+20 commanded-mm` release escape before character classification,
- released-character OCR after the key has been released,
- segmented retract back to command-space `Z=0`,
- a complete hardware result with controller state `SUCCEEDED` and outcome
  `SUCCESS` for target `G`.

The latest successful G checkpoint produced a strong SIDE event on the first
changed frame (`novel_pixels ~= 765`, largest component ~= 765 px), sent the
release command about 18 ms after that captured frame, preserved the latest-sent
XY exactly, and produced one stable new-character component. Released-character
OCR voted `G` in 3/5 frames and the controller completed with `SUCCESS`.

Hardware diagnostics also confirmed the Phase 3 command-space lesson under
contact load: requested Cartesian millimetres are not a claim of physical
millimetre-level TCP accuracy. During the successful `+20 commanded-mm` release,
the arm physically moved enough to cross the keyboard release point, but
`Present_Position` did not converge to the full `Goal_Position` before becoming
motion-stable. This low-level Goal/Present tracking behavior is recorded as a
future servo/contact-control optimization rather than a Phase 5 blocker.

The remaining Phase 5 work is therefore narrow:

- validate the same deterministic loop on at least two additional letter keys,
- do not tune `p_tip`, `J_cmd`, SIDE thresholds, tracking, OCR, or release depth
  per key,
- keep the operator-recovery path after autonomous completion; command-space
  `Z=0` is the fixed local anchor level, not the robot's physical home pose,
- then freeze the Phase 5 interfaces before collecting the Phase 6 ACT dataset.

Phase 3 remains the hardware-accepted deterministic motion foundation. Phase 4
remains the accepted conservative line-OCR semantic verifier. Phase 5 extends
those checkpoints rather than replacing them: WRIST remains the XY authority,
SIDE remains the physical outcome authority, and commanded depth remains a
motion/safety variable rather than a success detector.

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

A first `servo_ready` rule can require:

``` text
correct target label
AND confidence >= threshold
AND key size >= threshold
AND complete usable target observation
AND enough image-boundary margin for correction
AND stable for N consecutive fresh frames
```

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

After the local deterministic loop is reliable, collect
target-conditioned demonstrations.

Recommended policy observations:

``` text
observation.top
observation.wrist
robot joint state
target-key condition
```

Recommended action:

``` text
SO-101 joint command
```

Do not include the SIDE/SCREEN camera in the initial policy input.

### Demonstration endpoint

Teleoperation demonstrations should end at a **servo-ready viewpoint**,
not at an arbitrary pose and not necessarily at physical contact.

Good endpoints have:

- target visible,
- target recognizable,
- useful target pixel size,
- limited tool occlusion,
- safe press clearance.

This teaches ACT to hand the problem to classical perception rather than
to solve the entire contact task itself.

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
| Phase 5  | Deterministic Local Single-Key Closed Loop          | **IN PROGRESS — G CLEAN PASS; CROSS-KEY ACCEPTANCE PENDING** |
| Phase 6  | Target-Conditioned ACT Dataset                      | Not Started |
| Phase 7  | ACT Coarse Policy                                   | Not Started |
| Phase 8  | ACT + Visual Servo Handoff                          | Not Started |
| Phase 9  | Multi-Key Typing                                    | Not Started |
| Phase 10 | Automatic Recovery                                  | Not Started |
| Phase 11 | Controlled Generalization and Ablations             | Not Started |

A phase is complete only after its acceptance criteria have been validated on
the intended system.

## Current Phase

``` text
Phase 5 — Deterministic Local Single-Key Closed Loop
```

Current checkpoint:

``` text
Phase 1 camera/software foundation                 ✓ COMPLETED
        ↓
Phase 2 wrist perception                           ✓ COMPLETED
        ↓
Phase 3 fixed Goal anchor + WRIST visual servo     ✓ COMPLETED
        ↓
Phase 4 line OCR + four-state verifier             ✓ COMPLETED
        ↓
Phase 5 reusable fixed-anchor XYZ/planner          ✓
        ↓
Phase 5 press supervisor / iterative 2 mm descent  ✓
        ↓
low-Z semantic/geometry tracking                   ✓
        ↓
SIDE cursor-aware fast change detector             ✓
        ↓
strong glyph fast path: first-frame latch          ✓
ordinary event path: 2-frame confirmation          ✓
        ↓
latest-sent command-state ownership                ✓
stale event-interrupt XY rollback                  ✓ FIXED
        ↓
one +20 commanded-mm upward release                ✓
        ↓
released single-character OCR                      ✓
        ↓
segmented retract to command-space Z=0             ✓
        ↓
controller SUCCEEDED / outcome SUCCESS on G        ✓
        ↓
Goal/Present under-tracking under contact load     ↪ DEFERRED LOW-LEVEL OPTIMIZATION
        ↓
repeat on >=2 additional letters                   ← CURRENT ACCEPTANCE WORK
```

The current G checkpoint validates the intended architecture rather than
millimetre-level SO-101 TCP accuracy. The release command only needs to cross
the physical keyboard release point reliably; `Goal_Position == Present_Position`
is not a Phase 5 success requirement.

The Phase 5 code being committed is intended implementation code, not temporary
patch code:

- `src/so101_typing/control/fixed_anchor_xyz.py`
- `src/so101_typing/perception/screen_change.py`
- `scripts/validate_phase5_single_key.py`
- focused regression tests for latest-sent state ownership and the strong-event
  fast path.

Temporary patch installers, backup directories, generated debug images, and
one-off servo telemetry files under `artifacts/` are not runtime source and
should not be committed.

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

## Phase 5 — Deterministic Local Single-Key Closed Loop — **IN PROGRESS — G CLEAN PASS; CROSS-KEY ACCEPTANCE PENDING**

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

### Acceptance still pending

Phase 5 is not complete yet. Before acceptance:

- run the same deterministic loop on at least two additional letter keys;
- do not introduce per-key `p_tip`, `J_cmd`, OCR, tracking, SIDE-event, or
  press/release tuning;
- retain the current G parameters unless cross-key evidence exposes a generic
  architecture issue;
- keep `scripts/validate_phase3_xyz.py` as the Phase 3 hardware regression
  validator and `scripts/validate_phase5_single_key.py` as the Phase 5 hardware
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

## Phase 6 — Target-Conditioned ACT Dataset

Goal:

> Collect demonstrations that move the arm from valid starts to
> perception-aware handoff poses.

Record:

- TOP RGB,
- WRIST RGB,
- joint state,
- target key,
- action,
- timestamps.

The target condition must be part of every episode’s data schema.

Do not include screen-camera pixels in the ACT observation.

------------------------------------------------------------------------

## Phase 7 — ACT Coarse Policy

Goal:

> Given TOP + WRIST + robot state + requested key, move to a servo-ready
> local viewpoint.

Evaluate:

- target-neighborhood arrival rate,
- target acquisition rate after ACT,
- servo-ready rate,
- target visibility,
- target pixel size,
- occlusion rate,
- time to handoff.

ACT does not need to press the key.

------------------------------------------------------------------------

## Phase 8 — ACT + Visual Servo Handoff

Goal:

> Integrate the learned and deterministic controllers without command
> overlap.

Pipeline:

``` text
ACT_APPROACH
     ↓
target perception
     ↓
servo_ready
     ↓
flush/reset ACT
     ↓
SERVO_ALIGN
     ↓
small cumulative DESCEND_STEP
     ↓
STOP / REOBSERVE / SAME-Z REALIGN
     ↓
SIDE watcher
     ├── no event + confirmed NO_CHANGE → next descent step
     └── persistent screen change → EVENT LATCH → RELEASE
                                      ↓
                                released-char verify
                                      ↓
                                retract / supervisor
```

Acceptance must explicitly test:

- no stale ACT commands after handoff,
- deterministic controller ownership,
- target reacquisition behavior,
- handoff success rate,
- end-to-end single-key success rate.

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
coarse approach success rate
servo-ready handoff rate
target visibility after ACT
target pixel size after ACT
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
│   ├── collect_act_data.py
│   ├── train_act.py
│   └── run_typing_demo.py
│
├── tests/
│   ├── test_screen_ocr.py
│   ├── test_screen_line_ocr.py
│   ├── test_screen_verification.py
│   ├── test_state_machine.py
│   ├── test_recovery.py
│   ├── test_target_encoding.py
│   └── test_controller_ownership.py
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

The deterministic local portion of the first milestone has now been demonstrated
for target `G` from a safe manually positioned local pose. The **full hybrid
ACT+deterministic milestone** remains future work.

Current achieved deterministic behavior:

``` text
Input: G
   ↓
WRIST visually recognizes G
   ↓
fixed Goal-space anchor
   ↓
WRIST visual servo + similarity geometry fallback
   ↓
iterative 2 mm cumulative Z descent
   ↓
SIDE strong/confirmed screen-change event
   ↓
immediate +20 commanded-mm release escape
   ↓
released-character OCR
   ↓
CONFIRMED_SUCCESS
   ↓
segmented retract to command-space Z=0
   ↓
controller SUCCEEDED
```

Remaining work before the complete first hybrid milestone:

``` text
ACT uses TOP + WRIST + state + target=G
        ↓
ACT creates the same kind of servo-ready local viewpoint
        ↓
perception-triggered handoff preserves existing Goal_Position
        ↓
run the already-validated deterministic local loop
```

Nothing beyond this is required to prove the primary ACT-to-deterministic
architecture. Multi-character typing then becomes repeated execution plus
supervisor-driven recovery.

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
