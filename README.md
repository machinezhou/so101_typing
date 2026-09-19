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
7.  once alignment is stable, advance through bounded cumulative Z command
    levels relative to the same fixed command-space anchor, stopping after every
    level to reacquire vision and realign XY at the stopped Z level if needed,
8.  observe the MacBook screen through an independent camera after each
    stopped descent level,
9.  stop further descent as soon as the intended character is independently
    confirmed, then retract,
10. recover automatically from wrong key presses or bounded press failures,
11. repeat until the requested string is correct.
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

The project is no longer at the mechanical-feasibility stage. Phases 0–4 are
accepted; Phase 5 is the current implementation phase.

The current physical setup has already demonstrated:

- a calibrated SO-101 follower/leader setup,

- stable teleoperation,

- a previous ACT block-manipulation demo,

- a pencil mechanically attached to the gripper with cable ties,

- successful physical MacBook key presses using that tool,

- three simultaneous camera streams,

- a fixed MacBook and robot workspace.
  Therefore, the remaining engineering risk has shifted away from *“can the
  SO-101 physically press a key?”* and from the basic local XYZ control primitive
  toward:

- end-to-end integration of the accepted screen verifier with bounded press depth
  and retract behavior,

- robust ACT-to-deterministic-controller handoff using the validated Goal-space
  command-anchor contract,

- runtime ownership and automatic recovery,

- reproducible data collection and evaluation.

Phase 3 is hardware-accepted. The deterministic WRIST-controlled primitive
has demonstrated direct tool-tip calibration, dead-zone-aware fixed-anchor XY
visual servoing, geometry-based target tracking through glyph occlusion, and
cumulative staged Z with same-level XY recovery. Phase 4 is also accepted: the
fixed SIDE view now produces a conservative four-state screen outcome through
line-level Tesseract OCR, semantic prefix/expected-character comparison, and
multi-frame voting. The current project phase is **Phase 5 — Deterministic Local
Single-Key Closed Loop**.

------------------------------------------------------------------------

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
screen observation to decide whether the requested key has actually produced
the intended character. Commanded Z depth alone is never treated as proof of
success.

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

The SIDE camera is positioned as close to the MacBook screen as the
robot workspace safely permits. A view around 45° is acceptable.

It answers:

> **Did the physical action actually produce the intended result?**

Because the MacBook and camera are fixed, perspective distortion can be
removed with a one-time screen homography:

``` text
SIDE raw frame
      ↓
fixed screen quadrilateral
      ↓
perspective rectification
      ↓
canonical screen image
      ↓
fixed text ROI
      ↓
OCR / text recognition
      ↓
observed string
```

The SIDE/SCREEN stream should remain outside the initial ACT
observation.

------------------------------------------------------------------------

# System Architecture

The primary architecture is:

``` text
                         TARGET TEXT
                           "ROBOT"
                              │
                              ▼
                     ┌─────────────────┐
                     │ Task Supervisor │
                     └────────┬────────┘
                              │
                       target key = R
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               │
      target-key encoding                     │
              │                               │
       TOP RGB + WRIST RGB                    │
       + robot joint state                    │
              │                               │
              ▼                               │
          ┌───────┐                           │
          │  ACT  │                           │
          └───┬───┘                           │
              │ coarse joint actions          │
              ▼                               │
       perception-aware                       │
        approach region                       │
              │                               │
              ▼                               │
     WRIST KEY PERCEPTION                     │
   keycaps → glyphs → target                  │
              │                               │
        target acquired?                      │
              │ yes                           │
              ▼                               │
        CONTROLLER HANDOFF                    │
      stop/reset ACT execution                │
      wait until motion-stable                │
      latch existing Goal_Position            │
              │                               │
              ▼                               │
       VISUAL SERVO (XY)                      │
   target center → pencil-tip pixel           │
              │                               │
        alignment stable?                     │
              │ yes                           │
              ▼                               │
     ADVANCE CUMULATIVE Z LEVEL               │
       (hold current XY command)              │
              │                               │
        stop + settle                         │
              │                               │
       fresh WRIST observation                │
        ├─ misaligned → XY realign at fixed Z │
        └─ aligned ───────────────┐             │
                                 ▼             │
                          SCREEN CAMERA        │
                                 │             │
                       rectify + OCR           │
                                 │             │
        success / no-change / wrong / uncertain
                              │               │
                              └───────────────►│
                                      Task Supervisor
```

The important architectural property is **controller ownership and command-space continuity**:

- ACT owns coarse motion only until a perception-ready handoff.
- The handoff waits for the follower to become motion-stable and then preserves
  the existing servo `Goal_Position` as the deterministic command-space anchor.
- After handoff, the deterministic staged controller owns cumulative XY alignment
  and bounded cumulative Z-level changes; ACT does not resume during the press
  attempt.
- A downward level transition changes cumulative Z while holding cumulative XY
  fixed. Lateral re-alignment is performed only after stop/settle at that fixed Z
  level.
- Screen perception never commands the robot directly, but its confirmed result
  determines whether further descent is allowed.
- The supervisor is the only module that changes high-level task state.

------------------------------------------------------------------------

# Runtime State Machine

The runtime should be implemented as an explicit state machine rather
than a loose sequence of function calls.

``` text
IDLE
  │
  ▼
SET_TARGET
  │
  ▼
ACT_APPROACH
  │
  ├── target not visible / not ready ───────► continue ACT
  │
  ▼
TARGET_ACQUIRED
  │
  ▼
HANDOFF
  │
  ├── stop ACT
  │
  ├── clear pending action chunk
  │
  ├── wait until follower motion is stable
  │
  ├── preserve existing Goal_Position as command anchor
  │
  ├── refresh Present_Position for diagnostics/safety
  │
  └── acquire fresh wrist frame
  │
  ▼
SERVO_ALIGN
  │
  ├── target lost / stale ──────────────────► RECOVERY
  │
  ├── timeout / correction budget ──────────► RECOVERY
  │
  ▼
ALIGNED_AT_LEVEL
  │
  │  initial level, or previous VERIFY_LEVEL authorized another step
  ▼
DESCEND_STEP
  │
  ▼
SETTLE_AND_REOBSERVE
  │
  ├── alignment drifted ────────────────────► REALIGN_AFTER_STEP
  │                                           │
  │                                           ├── target lost / stale ─► RECOVERY
  │                                           └── aligned ─────────────► VERIFY_LEVEL
  │
  └── alignment still valid ────────────────► VERIFY_LEVEL
                                              │
                                              ├── CONFIRMED_SUCCESS ───► RETRACT ─► NEXT_TARGET
                                              ├── CONFIRMED_NO_CHANGE ─► DESCEND_STEP
                                              ├── CONFIRMED_WRONG ─────► RETRACT ─► RECOVERY
                                              ├── UNCERTAIN ───────────► HOLD / REOBSERVE
                                              └── safety bound ────────► RETRACT ─► RECOVERY

DONE
```

`TARGET_ACQUIRED` and `ALIGNED_AT_LEVEL` are intentionally different states:

- `TARGET_ACQUIRED`: the system knows where the requested key is in the wrist
  image and has enough image margin for deterministic correction.
- `ALIGNED_AT_LEVEL`: the target-key center and calibrated pencil-tip pixel are
  stably aligned at the current stopped Z level. This authorizes **one** bounded
  cumulative Z-level advance from the fixed Goal-space anchor, not continuous
  downward motion.

Every Z-level change invalidates the previous alignment acceptance. The
controller must stop, settle, acquire fresh observations, and realign at the same
stopped Z level if necessary. If
realignment is needed after a Z step, the controller must still verify the
screen at that same stopped level before any further descent; successful
realignment must not accidentally authorize an extra Z step. A press succeeds
only on independent screen confirmation; a configured maximum descent remains
a hard failure/safety bound.

------------------------------------------------------------------------

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

The press stage should not be learned in V0. It is a bounded staged descent from
the same fixed Goal-space anchor, not one precomputed downward stroke and not a
sequence that relatches from measured state:

``` text
stable XY alignment at current level
        ↓
advance cumulative Z level
(hold cumulative XY fixed)
        ↓
stop + settle
        ↓
fresh WRIST observation
        ↓
realign XY at this same Z level if needed
        ↓
independent SIDE/screen verification
        ↓
CONFIRMED_SUCCESS ? retract : next authorized Z level
```

The current validated Phase 3 primitive used cumulative commanded Z levels of
`-3 mm` and `-6 mm` from the original Goal-space anchor. These are command-space
validation levels, not measured TCP displacement and not the final press-depth
policy. Phase 5 will let screen verification authorize any further bounded level.

The controller must define:

- bounded cumulative Z command levels,
- maximum cumulative downward command budget,
- maximum XY correction and total Cartesian command budget,
- speed/acceleration limits,
- settle/fresh-frame requirements,
- workspace/joint safety bounds,
- timeout behavior,
- abort/retract behavior.

A completed motion command does **not** mean the arm is physically settled. The
SO-101 may show a short post-motion wobble, non-zero servo residual, and XYZ
dead zone/backlash. Observations captured during the settling window must not
authorize a correction or another Z level.

The important dead-zone rule is:

> **Do not interpret a small commanded XYZ change with little measured motion as
> controller failure, and do not relatch the command origin from
> `Present_Position`.**

Instead, keep one existing-`Goal_Position` anchor and advance bounded cumulative
commands. `Present_Position` and measured FK remain diagnostic/safety signals;
they are not press-success detectors.

Phase 3 hardware validation demonstrated the full local primitive:

``` text
XY stable at 1.79 px
        ↓
cumulative Z = -3 mm
        ↓
XY drift detected and re-aligned at the same Z level
        ↓
XY = 4.26 px
        ↓
cumulative Z = -6 mm
        ↓
XY = 2.88 px
        ↓
staged XYZ primitive complete
```

The pencil/tool must never advance to a deeper Z level if the current visual
alignment is invalid. Commanded descent depth is a safety/budget quantity, not a
success detector. Further descent stops immediately when screen verification
returns `CONFIRMED_SUCCESS`; if success is never confirmed before the hard
maximum descent/safety bound, the attempt fails and retracts.

# Screen Perception and Verification

The screen camera is an independent observer.

Because the SIDE camera may view the screen at approximately 45°, V0
should explicitly rectify the screen before OCR.

## Calibration

Record the four screen corners once:

``` text
raw SIDE frame
      ↓
screen quadrilateral
      ↓
homography H_screen
      ↓
canonical screen
```

Then use a fixed typing ROI.

## Verification states

Screen verification should not be binary.

Use:

``` text
CONFIRMED_SUCCESS
CONFIRMED_NO_CHANGE
CONFIRMED_WRONG
UNCERTAIN
```

Verification is relative to the screen text confirmed immediately before the
current key attempt. For example, if the confirmed pre-press prefix is `ROB`
and the current requested key is `O`, the expected post-press prefix is `ROBO`.

`CONFIRMED_NO_CHANGE` means fresh SIDE observations still stably match the
confirmed pre-press text, so the stopped descent level has not yet produced the
expected character. If all WRIST/safety gates still pass, another bounded Z
step may be attempted. `UNCERTAIN` is different: an OCR fluctuation must not
trigger either further descent or destructive recovery such as `BACKSPACE`
until the observation is resolved.

Example:

``` text
confirmed before attempt: ROB
requested key:            O
expected after success:   ROBO

observed text:             ROB   -> CONFIRMED_NO_CHANGE
observed text:             ROBO  -> CONFIRMED_SUCCESS
other stable text:                -> CONFIRMED_WRONG
unstable / low-confidence OCR:    -> UNCERTAIN
```

If the screen result is uncertain:

``` text
wait / reacquire / OCR again
```

rather than modifying the typed string immediately.

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

| Phase    | Description                                         | Status                    |
|----------|-----------------------------------------------------|---------------------------|
| Phase 0  | Mechanical Feasibility                              | **Completed**             |
| Phase 1  | Freeze Camera Geometry and Build Camera Sanity Tool | **Completed**             |
| Phase 2  | Wrist Keycap Detection and Glyph Recognition        | **Completed**             |
| Phase 3  | Tool Reference and Visual Servo                     | **Completed**             |
| Phase 4  | Screen Rectification and Verification               | **Completed**             |
| Phase 5  | Deterministic Local Single-Key Closed Loop          | **IN PROGRESS — CURRENT** |
| Phase 6  | Target-Conditioned ACT Dataset                      | Not Started               |
| Phase 7  | ACT Coarse Policy                                   | Not Started               |
| Phase 8  | ACT + Visual Servo Handoff                          | Not Started               |
| Phase 9  | Multi-Key Typing                                    | Not Started               |
| Phase 10 | Automatic Recovery                                  | Not Started               |
| Phase 11 | Controlled Generalization and Ablations             | Not Started               |

A phase is complete only after its acceptance criteria have been
validated on the intended system. Placeholder modules, diagnostic
components, or partial hardware tests do not by themselves complete a
phase.

## Current Phase

``` text
Phase 5 — Deterministic Local Single-Key Closed Loop
```

Current checkpoint:

``` text
Phase 1 camera/software foundation             ✓ COMPLETED
        ↓
Phase 2 wrist perception                         ✓ COMPLETED
        ↓
Phase 3 tool reference + visual servo             ✓ COMPLETED
        ↓
canonical J_cmd                                   ✓
        ↓
direct WRIST p_tip                                ✓ (307.0, 238.246)
        ↓
existing Goal_Position command anchor             ✓
        ↓
dead-zone-aware cumulative XY servo               ✓
        ↓
semantic lock + geometry tracking under occlusion ✓
        ↓
wide local capture: 61.89 px -> 4.56 px           ✓ demonstrated
        ↓
stable XY: 43.43 px -> 1.79 px (2 fresh frames)  ✓
        ↓
cumulative Z 0 -> -3 -> -6 mm                    ✓
        ↓
low-Z XY re-alignment at fixed Z                  ✓
        ↓
final staged-Z validation error 2.88 px           ✓
        ↓
PHASE 3 ACCEPTED
        ↓
SIDE homography + text ROI                         ✓ existing from Phase 1
        ↓
line-level Tesseract OCR                            ✓
        ↓
semantic prefix + expected-character comparison    ✓
        ↓
9-frame conservative voting                         ✓
        ↓
NO_CHANGE / SUCCESS / WRONG live validation        ✓ 9/9 each
        ↓
UNCERTAIN fail-safe behavior                        ✓ unit-tested
        ↓
PHASE 4 ACCEPTED
        ↓
PHASE 5 DETERMINISTIC SINGLE-KEY CLOSED LOOP      <-- CURRENT
```

## Phase 1 Completion Record

Phase 1 acceptance criteria have now been validated on the intended
three-camera system. Camera positions and the fixed MacBook geometry
should remain unchanged unless a later phase demonstrates a concrete
failure that requires recalibration.

### Stable three-camera baseline

All three cameras have been tested simultaneously with the intended
resolution, frame rate, and pixel format.

| Camera | OpenCV ID | Resolution | Target FPS | FOURCC | Measured FPS |
|--------|----------:|-----------:|-----------:|--------|-------------:|
| TOP    |         2 |    640x480 |         30 | MJPG   |        ~29.8 |
| WRIST  |         0 |    640x480 |         30 | YUYV   |        ~30.0 |
| SIDE   |         4 |    640x480 |         30 | YUYV   |        ~30.0 |

The validated 30-second simultaneous sanity run completed with zero
camera read errors and passed the FPS check:

``` text
TOP     29.80 FPS  errors=0
WRIST   29.99 FPS  errors=0
SIDE    29.99 FPS  errors=0
FPS CHECK: PASS
```

The camera adapter uses a continuously draining threaded reader and
retains only the latest application-level frame. Do not set
`cv2.CAP_PROP_BUFFERSIZE=1` for these cameras. Hardware testing showed
that this setting reduced TOP MJPG capture from approximately 30 FPS to
approximately 15 FPS even though the requested camera FPS remained 30.

The runtime frame contract remains active:

``` text
camera_name
frame_id
capture_timestamp
processing_timestamp
frame_age_ms
```

`capture_timestamp` is currently a local monotonic timestamp recorded
immediately after a successful OpenCV `read()` returns. It is not a
hardware sensor exposure timestamp.

### Reproducible camera controls

Camera controls required for stable operation are stored in the camera
configuration and applied automatically before OpenCV opens the device.

Current validated control strategy:

``` text
TOP
  auto_exposure = 1
  exposure_time_absolute = 200
  gain = 32

WRIST
  auto_exposure = 3
  exposure_dynamic_framerate = 0

SIDE
  no additional V4L2 control override currently required
```

A hardware persistence test intentionally changed TOP and WRIST device
controls to incorrect values before project startup. `camera_sanity.py`
restored the configured values automatically while maintaining the
validated simultaneous capture rates. Requested V4L2 controls are also
recorded in `artifacts/camera_sanity/report.json`.

### Final camera geometry

The final TOP view covers the keyboard, SO-101, and useful approach
workspace. The TOP mount was adjusted before final acceptance so that an
unwanted dynamic region at the upper-right of the previous framing is no
longer part of the intended observation. The accepted TOP geometry is
now frozen.

The WRIST view remains accepted as the Phase 1 handoff geometry. At a
representative pre-contact pose it contains multiple local keycaps with
readable glyph detail, while preserving room for additional motion
toward the keyboard. Tool-reference, visual-servo, and press validation
remain later-phase responsibilities.

The SIDE camera position and MacBook screen angle are frozen. The full
relevant display region remains available for the fixed homography, and
the rectified output contains readable screen text.

### SIDE screen calibration

The SIDE calibration is now configured in
`calibration/screen_homography.json`:

``` json
{
  "source_points": [
    [40.0, 84.0],
    [463.0, 60.0],
    [431.0, 435.0],
    [25.0, 340.0]
  ],
  "output_size": [1280, 800],
  "text_roi": [304, 67, 960, 694]
}
```

The calibration path has been validated both offline and through the
live camera pipeline:

``` text
SIDE raw
    ↓
fixed screen quadrilateral
    ↓
homography
    ↓
1280x800 canonical screen
    ↓
fixed 960x694 text ROI
    ↓
readable screen content
```

`ScreenCalibration` now validates source geometry and ROI bounds and
supports calibration save/load. `scripts/calibrate_screen.py` provides a
browser-based calibration workflow compatible with the project's
headless OpenCV environment.

### Optional TOP screen-mask calibration

The TOP display mask is retained as an optional screen-leakage / ablation
capability, not as mandatory preprocessing for the default ACT path.
Default ACT experiments should therefore begin with raw TOP unless a
leakage experiment provides evidence that masking is required.

The accepted optional mask is stored in
`calibration/top_screen_mask.json`:

``` json
{
  "polygon": [
    [47, 351],
    [144, 94],
    [258, 165],
    [204, 369]
  ],
  "brightness_threshold": 245
}
```

The calibration preview confirms that the visible MacBook display is
removed while the keyboard, robot, and useful workspace remain visible.
The calibration artifact reported approximately:

``` text
masked_fraction               0.11379
screen_pixel_count            34957
bright_fraction_inside_screen 0.03456
mean_inside_screen            242.35
std_inside_screen             6.87
```

`TopMaskConfig` now validates and persists the fixed polygon, and
`scripts/calibrate_top_mask.py` provides the corresponding browser-based
calibration workflow. The saved mask remains available for the later
`TOP raw` vs `TOP screen-masked` ablation.

### Phase 1 software validation

The SIDE calibration changes add unit coverage for uncalibrated loading,
valid rectification/cropping, invalid quadrilaterals, ROI bounds, and
save/load round trips. The TOP mask changes add coverage for masking and
statistics, passthrough behavior, polygon validation, frame bounds,
save/load round trips, and brightness-threshold validation.

The 11 calibration-focused unit tests supplied with the Phase 1
calibration checkpoints pass together.

## Phase 1 Acceptance

The Development Roadmap Phase 1 acceptance criteria are satisfied:

``` text
three stable ~30 FPS camera streams
        +
zero read errors in the validated simultaneous run
        +
accepted and frozen camera geometry
        +
TOP useful robot/keyboard workspace
        +
WRIST usable local keycap/glyph view
        +
SIDE readable rectified screen
        +
fixed SIDE text ROI
        +
frame timing metadata
        +
reproducible camera controls
        +
validated calibration files and calibration tools
        ↓
PHASE 1 COMPLETED
        ↓
PHASE 2 COMPLETED
        ↓
PHASE 3 COMPLETED
        ↓
PHASE 4 COMPLETED
        ↓
PHASE 5 IN PROGRESS — CURRENT
```

## Phase 2 Completion Record

Phase 2 has now been validated as the V0 wrist-perception checkpoint.
Key identity is inferred from visible glyph appearance rather than from a
pre-programmed keyboard row/column identity.

### Wrist perception dataset

The completed dataset contains:

``` text
24 independent WRIST source frames
333 labeled real glyph crops
26 / 26 A-Z classes covered
```

The first seed round contained 93 glyphs from 8 source frames. A second
round added 240 high-confidence glyph crops from 16 additional WRIST
poses, including stronger viewpoint variation, bottom-region views, and
tool-occlusion cases.

The second-round manifest uses normalized image-coordinate anchors and
nearest-candidate matching rather than relying on raw contour candidate
indices. This avoids brittle label association when small OpenCV contour
count differences occur across environments.

### Classifier selection

The initial strict whole-source-frame held-out comparison established the
baseline:

``` text
Template Matching      58 / 92 = 63.0%   (1 skipped class fold)
HOG + Linear SVM       78 / 92 = 84.8%   (1 skipped class fold)
```

After adding the second-round viewpoints, the combined dataset was
re-evaluated with leave-one-source-frame-out grouping:

``` text
HOG + Linear SVM      322 / 333 = 96.7%
skipped = 0
```

This exceeded the approximately 95% decision threshold used for the V0
classifier choice. HOG + Linear SVM is therefore the accepted Phase 2
classifier path; Tiny CNN is not required for the current fixed-environment
V0 checkpoint.

The remaining grouped-held-out errors were concentrated mainly among
visually similar classes such as T/Y, P/F, L/I, and M/H rather than a
broad failure of the pipeline.

### Runtime perception wiring

The offline classifier has been integrated into the runtime perception
path:

``` text
WRIST frame
    ↓
keycap candidates
    ↓
per-key rectification
    ↓
glyph preprocessing
    ↓
persisted HOG + Linear SVM
    ↓
confidence / conservative rejection
    ↓
target-letter filtering
    ↓
TargetObservation
```

The completed runtime wiring includes:

- persisted classifier loading,
- conservative unknown/reject handling,
- A-Z target selection,
- integration with `TargetObservation`,
- validation across the collected multi-view WRIST data.

The validated runtime wiring result is:

``` text
329 / 333 = 98.8% runtime wiring recall
```

This number is intentionally reported as runtime wiring recall, not as
held-out classification accuracy. The grouped held-out classifier result
remains 322 / 333 = 96.7%.

## Phase 2 Acceptance

The Phase 2 checkpoint is accepted for V0 because the system now has:

``` text
appearance-based key identity
        +
real multi-view WRIST data
        +
A-Z coverage
        +
96.7% grouped held-out HOG + SVM accuracy
        +
persisted runtime classifier
        +
conservative rejection
        +
A-Z target filtering
        +
TargetObservation runtime integration
        +
98.8% runtime wiring recall
        ↓
PHASE 2 COMPLETED
        ↓
PHASE 3 COMPLETED
        ↓
PHASE 4 COMPLETED
        ↓
PHASE 5 IN PROGRESS — CURRENT
```

## Phase 3 Completion Record

Phase 3 is accepted. It validated the deterministic WRIST-controlled primitive
from a safe teleoperated pose without requiring ACT or screen-confirmed keypress
success.

### Accepted calibration and perception/control references

- The direct physical WRIST pencil-tip reference is now calibrated in
  `calibration/tool_reference.json` at approximately `(307.0, 238.246) px`,
  using five direct samples with approximately 1.62 px maximum radial deviation.
- H3.2 remains the canonical command-space image Jacobian checkpoint:

  ``` text
  J_cmd [px/commanded-mm] =
  [[-2.386845397949219,   -0.41411895751953126],
   [ 0.29467163085937503,  2.87500000000000000]]
  ```

- H3.2 acceptance metrics remain **residual RMS = 3.256 px**, **condition
  number = 1.391**, **X opposition = -0.981**, and **Y opposition = -0.931**.
- The canonical `calibration/image_jacobian.json` keeps
  `input_semantics = requested_cartesian_delta`. Measured-joint FK remains
  diagnostic-only for physical accuracy.

### Dead zone / preload finding and command-anchor correction

Physical integration exposed a critical SO-101 behavior: the arm can be
motion-stable with a non-zero `Goal_Position - Present_Position` residual. XYZ
also exhibit dead zone/backlash/compliance, so small command increments can be
absorbed without immediate visible motion.

A zero-delta handoff experiment isolated the failure mode:

- rebasing a nominal zero Cartesian command on `Present_Position` caused a
  visible WRIST jump even though the planned Cartesian delta was zero;
- waiting for the arm to settle did not remove that effect;
- resending the **existing `Goal_Position` unchanged** produced zero Goal change,
  zero measured joint change after settle, and zero WRIST target shift.

The accepted command contract is therefore:

``` text
manual / ACT motion stops
        ↓
wait for physical motion stability
        ↓
read existing Goal_Position once
        ↓
fixed command-space anchor
        ↓
cumulative XYZ commands relative to that anchor
        ↓
Present_Position / FK = diagnostics + safety only
        ↓
WRIST outcome = XY success authority
```

Do **not** relatch the command origin from `Present_Position` after a small step,
and do not treat a tiny command with little measured motion as evidence that the
controller direction is wrong.

### Accepted XY visual-servo behavior

The final Phase 3 XY controller uses bounded cumulative command-space correction
with a 5 mm per-step limit, fresh WRIST observations, and whole-keyboard geometry
tracking when the already-identified glyph becomes occluded by the pencil.

Hardware validation demonstrated:

``` text
61.89 px initial error
        ↓
bounded cumulative XY control
        ↓
4.56 px final error
```

A separate staged-XYZ validation converged:

``` text
43.43 px initial error
        ↓
1.79 px stable alignment
(2/2 fresh frames)
```

The old 20 px debug capture gate is retired. The demonstrated deterministic
capture range is at least about 62 px in the tested fixed setup. This does not
make 62 px a universal hard threshold; it establishes that ACT does not need to
place the tool within 20 px before handoff.

### Accepted occlusion handling

Near alignment the pencil can obscure the target glyph. Immediate target-loss
abort was therefore replaced with a guarded semantic-lock/geometry-tracking
strategy:

- glyph recognition establishes the target identity;
- if the glyph later disappears, multi-key keyboard geometry estimates image
  translation and propagates the already-established target center;
- geometry tracking must pass strict match/inlier/residual gates;
- a poor geometry frame is retried within the capture timeout rather than being
  accepted or immediately treated as controller failure.

Geometry tracking never chooses a new key identity.

### Accepted cumulative XYZ staged primitive

Phase 3 completed the same-anchor staged primitive using cumulative command-space
Z levels `0 -> -3 -> -6 mm`. These are commanded-mm levels, not measured TCP
displacement claims.

The accepted hardware run was:

``` text
XY stable = 1.79 px
        ↓
cumulative Z = -3 mm
        ↓
XY drift = 7.26 px
        ↓
three same-Z cumulative XY corrections
        ↓
XY restored = 4.26 px
        ↓
cumulative Z = -6 mm
        ↓
XY = 2.88 px
        ↓
Z stage status = complete
```

This validates fixed-anchor cumulative XYZ, stop/settle/reobserve behavior, and
same-level XY recovery. It is **not** a keypress-success claim. Screen-confirmed
success and any further bounded descent belong to Phase 5 after Phase 4 screen
verification is available.

### Phase 3 implementation notes / pitfalls

These lessons are now part of the accepted runtime design:

- Do **not** treat FK -> IK -> FK consistency as physical millimetre accuracy.
- Do **not** use `Present_Position` as a new command anchor during deterministic
  runtime control; preserve the existing Goal-space preload.
- X, Y, and Z all require dead-zone-aware cumulative command semantics. A small
  no-motion result is not a direction/failure test.
- Do not repeatedly issue tiny reset-style corrections that restart inside the
  dead zone. Use bounded cumulative commands from the fixed Goal anchor.
- `Present_Position` and FK may be used for motion stability, diagnostics,
  workspace/joint safety, and IK seeding, but not as the authority for XY
  convergence or press success.
- The calibrated `J_cmd` is a command-space mapping:
  `requested Cartesian XY delta -> observed WRIST pixel delta`.
- Preserve strict separation between a Z-level change and an XY correction: hold
  cumulative XY fixed while advancing Z, then stop/settle and realign XY at the
  same fixed Z level if necessary.
- Target identity comes from glyph appearance. Geometry tracking is only a
  temporary propagation mechanism after semantic identity is already known.
- Do not chase sub-pixel error near convergence. The validated V0 control band is
  a few pixels; Phase 3 used 4 px / 2 fresh frames for initial alignment and a
  6 px Z-level handoff/recheck tolerance.
- Keep H3.2 as a saved calibration checkpoint unless fixed camera/tool/keyboard
  geometry changes or later closed-loop evidence contradicts it.

### Phase 3 hardware ownership and abort contract

Do **not** hand Phase 3 from a separate `lerobot-teleoperate` process to a
second robot process. `SO101Follower.connect()` performs follower
configuration with a torque-disabled section, so reconnecting while the arm is
already hovering above the keyboard would reintroduce an avoidable sag/handoff
risk.

The Phase 3 hardware process should own the follower and leader for the whole
session:

``` text
start with leader + follower at normal zero/home
        ↓
process connects follower once (safe rest pose)
        ↓
process connects leader
        ↓
in-process leader -> follower teleoperation
        ↓
operator moves to perception-safe hover
        ↓
press ENTER to end manual positioning
        ↓
wait until follower motion is stable
        ↓
latch existing Goal_Position as fixed command anchor
        ↓
autonomous bounded cumulative XY / staged-Z control
        ↓
in-process operator recovery teleoperation resumes
        ↓
operator returns to normal zero/home
        ↓
Ctrl+C
        ↓
normal LeRobot disconnect / torque-off
```

This keeps the follower connected across manual positioning and autonomous
motion, so the transition does not call `SO101Follower.connect()` again at the
hover pose.

Exit semantics remain deliberately separated:

- **normal deterministic-stage completion:** stop autonomous corrections and resume
  operator-controlled leader/follower teleoperation; the operator returns to
  the normal zero/home pose and then presses Ctrl+C for normal LeRobot
  disconnect/torque-off;
- **controlled failure** (`target_lost`, stale-frame timeout, correction
  budget, unexpected clipping, settle failure): stop autonomous corrections
  and enter the same operator recovery teleoperation when the hardware link is
  still healthy;
- **Ctrl+C during autonomy:** cancel further autonomous commands and enter
  operator recovery; Ctrl+C during recovery means the operator has selected a
  safe disconnect point;
- **physical emergency / broken communication:** do not attempt an automatic
  recovery trajectory. Use the physical stop/power procedure as appropriate.

Do not interpret "abort" as "blindly drive home". Returning home is an
operator-controlled recovery step after autonomous motion has stopped.

### H3.2 recalibration policy

The accepted H3.2 result is now canonical. A future H3.2 recalibration should
only be run after a relevant camera/tool/keyboard geometry change or if later
closed-loop evidence shows that the local mapping is no longer valid.

When recalibration is actually required, preserve the accepted protocol:

``` text
teleoperate to perception-safe hover
        ↓
latch the existing Goal_Position as one fixed command-space anchor
        ↓
adaptive +X/-X/+Y/-Y conditioning
        ↓
conditioning convergence gate
        ↓
start a fresh formal paired sample set
        ↓
fit and review candidate J_cmd
        ↓
explicit promotion only after acceptance
```

Conditioning samples remain excluded from the Jacobian fit. The full
calibration writes a **candidate** under `artifacts/image_jacobian_calibration/`;
the canonical file is changed only by explicit review/promotion:

``` bash
python scripts/promote_image_jacobian.py --confirm-reviewed
```

Phase 3 is now accepted and frozen as the deterministic local motion checkpoint.
ACT remains outside this checkpoint, and screen-confirmed keypress completion
remains a Phase 5 integration task.

The accepted Phase 3 hardware logic is intentionally retained in
`scripts/validate_phase3_xyz.py` as a **hardware acceptance / regression
validator**, not as the final production runtime. Productionizing that validated
logic is deliberately deferred to Phase 5, after Phase 4 provides the independent
screen-verification authority needed by the real press loop. In particular,
Phase 5 will extract the fixed-Goal-anchor cumulative XYZ state, same-Z XY
realignment, ownership/handoff rules, and screen-authorized descent/retract
behavior into reusable `src/so101_typing/control/` and `runtime/` modules with new
unit tests. Legacy pre-Goal-anchor Phase 3 runners and the old fixed `-0.5 mm`
step-count staged abstraction are not part of the accepted runtime design and
should not be reused as the basis of Phase 5.

## Phase 4 Completion Record

Phase 4 is accepted as the independent screen-observation checkpoint. The fixed
SIDE camera, Phase 1 screen homography, and fixed text ROI were retained; no
screen recalibration was required.

The accepted V0 OCR / verification path is:

``` text
SIDE 640x480 fresh frame
        ↓
calibration/screen_homography.json
        ↓
1280x800 rectified screen
        ↓
fixed 960x694 typing ROI
        ↓
detect dark text-line bands
        ↓
per-line crop + 3x upscale + CLAHE
        ↓
Tesseract 5.3.4, OEM 1, PSM 7, English model
A-Z / 0-9 character whitelist
        ↓
compare OCR lines with confirmed prefix + expected character
        ↓
9 fresh-frame semantic voting, 60% confirmation threshold
        ↓
CONFIRMED_SUCCESS / CONFIRMED_NO_CHANGE /
CONFIRMED_WRONG / UNCERTAIN
```

Tesseract is used as the system OCR engine with its pretrained English model;
Phase 4 does **not** train a project-specific OCR network. The project deliberately
keeps OCR observation separate from semantic verification: ambiguous characters
such as `O/0` or `I/1` are not silently rewritten to the expected answer.

A whole-ROI / whole-text baseline was tested first. It produced readable text but
only `2/9` exact-string consensus on a fixed screen, so exact full-text equality
was rejected as the authority criterion. The accepted implementation instead
uses physical text-line localization, single-line OCR, tolerant confirmed-prefix
matching, the observed continuation character, and multi-frame voting. This also
makes harmless editor decorations such as the visible leading line number `1`
non-authoritative.

Controlled live acceptance used confirmed prefix `KEYPRESS`, expected character
`G`, and wrong-character probe `H`:

``` text
screen = KEYPRESS
    -> CONFIRMED_NO_CHANGE   9/9 votes

screen = KEYPRESSG
    -> CONFIRMED_SUCCESS     9/9 votes

screen = KEYPRESSH
    -> CONFIRMED_WRONG       9/9 votes
       observed wrong char = H
```

`UNCERTAIN` is the fail-safe result whenever no authoritative outcome reaches the
required vote threshold; disagreement and unstable wrong-character evidence are
covered by unit tests. The verifier therefore favors withholding authorization
over inventing a success or wrong-key claim.

Accepted Phase 4 implementation artifacts are intended to be:

- `src/so101_typing/perception/screen_ocr.py` — line detection, Tesseract OCR,
  normalization, and OCR observation contracts;
- `src/so101_typing/supervisor/verification.py` — four-state semantic verifier;
- `tests/test_screen_ocr.py`, `tests/test_screen_line_ocr.py`, and
  `tests/test_screen_verification.py` — offline OCR/verifier tests;
- `scripts/validate_phase4_screen_verification.py` — live SIDE-camera acceptance
  validator without robot motion.

The exploratory whole-ROI OCR probe and exact-string consensus validator are not
part of the accepted runtime architecture and may be removed after this checkpoint.

## Phase 5 Current Checkpoint

Phase 5 is now the active phase. Its job is to integrate the two independently
accepted deterministic subsystems without changing their authority boundaries:

``` text
Phase 3 WRIST controller
fixed existing Goal_Position anchor
+ cumulative XY/Z
+ same-Z XY recovery
        ↓
stop + settle + fresh observations
        ↓
Phase 4 SIDE verifier
CONFIRMED_NO_CHANGE / CONFIRMED_SUCCESS /
CONFIRMED_WRONG / UNCERTAIN
        ↓
press-depth authorization / retract / recovery
```

The first Phase 5 target is one deterministic local keypress from a safe local
pose. ACT remains outside this checkpoint. Screen verification is authoritative
for physical keypress success; commanded depth remains only a bounded motion
budget and safety quantity.

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

## Phase 5 — Deterministic Local Single-Key Closed Loop — **IN PROGRESS — CURRENT**


Goal:

> From a safe pose near a requested key, visually acquire, align, descend in
> bounded stages, stop when the screen independently confirms success, and
> retract.

Example:

``` text
target=G
   ↓
wrist recognizes G
   ↓
p_key -> p_tip XY alignment
   ↓
alignment stable
   ↓
advance one bounded cumulative Z level
   ↓
stop + settle + fresh WRIST observation
   ↓
realign if needed
   ↓
screen verify
   ├── CONFIRMED_NO_CHANGE → next bounded Z step
   ├── UNCERTAIN → hold / reobserve
   ├── CONFIRMED_WRONG → retract / recovery
   └── CONFIRMED_SUCCESS → retract / SUCCESS
```

This proves the local perception-action-verification loop before learned coarse
motion is introduced. The hard maximum cumulative descent is a safety/failure
bound, never a substitute for screen-confirmed success.

### Phase 5 implementation migration / productionization

Phase 5 is also the point where the accepted Phase 3 hardware logic is moved out
of the validation script and into the production architecture. Do **not** revive
or adapt the retired pre-Goal-anchor staged controller. The implementation should
be rebuilt around the hardware-accepted contract:

``` text
motion-stable handoff
        ↓
read existing Goal_Position once
        ↓
fixed command-space anchor
        ↓
cumulative XY / Z command state
        ↓
stop + fresh WRIST observation after each command/level
        ↓
same-Z XY realignment when needed
        ↓
SIDE verification authorizes next Z level / retract / failure
```

Required Phase 5 engineering work:

- extract the validated fixed-anchor cumulative XYZ command state from
  `scripts/validate_phase3_xyz.py` into reusable control/runtime modules;
- preserve the existing `Goal_Position` preload across the full local press
  attempt; `Present_Position` remains diagnostics/safety/IK-seed state and must
  never silently become a new command origin;
- keep XYZ dead-zone/backlash semantics explicit: a small no-motion response does
  not trigger rebasing, direction reversal, or reset-style micro-steps;
- preserve semantic target identity through temporary near-contact glyph
  occlusion using the guarded geometry fallback already validated in Phase 3;
- make every Z-level transition cumulative from the original Goal-space anchor,
  hold XY fixed during the transition, then allow XY correction only after
  stop/settle at that same Z level;
- integrate the Phase 4 verifier so `CONFIRMED_SUCCESS` stops further descent,
  `CONFIRMED_NO_CHANGE` may authorize the next bounded Z level, `UNCERTAIN` holds
  and reobserves, and `CONFIRMED_WRONG` retracts/fails;
- implement bounded retract and operator-safe failure behavior without blind
  automatic homing.

New tests should cover at least:

- fixed Goal-anchor creation and zero-delta preservation;
- cumulative XYZ state without `Present_Position` rebasing;
- dead-zone-safe accumulation across multiple commands;
- Z-level advance with XY held fixed;
- same-Z XY realignment preserving cumulative Z;
- semantic-lock geometry fallback not selecting a new key identity;
- no further downward command after `CONFIRMED_SUCCESS`, `CONFIRMED_WRONG`, or an
  exhausted safety budget;
- `UNCERTAIN` producing hold/reobserve rather than descent;
- controller ownership and stale-command rejection at the deterministic handoff.

`scripts/validate_phase3_xyz.py` should remain available as a hardware regression
validator after this extraction; it should not become the production typing
runtime itself.

Phase 5 should also repeat the deterministic local loop on at least two
additional letter keys without per-key `p_tip` values or per-key Jacobians. This
absorbs the cross-key transfer sanity check that was removed from Phase 3 after
the control architecture changed during hardware validation.

------------------------------------------------------------------------

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
DESCEND_STEP
     ↓
STOP / REOBSERVE / REALIGN
     ↓
SCREEN_VERIFY
     ├── NO_CHANGE → next bounded descent level
     ├── UNCERTAIN → hold / reobserve
     └── SUCCESS / WRONG / safety bound → retract / supervisor
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
maximum cumulative downward motion per press attempt
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
8.  failed/uncertain verification never causes an unbounded descent or retry loop.

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
│       │   ├── screen_rectify.py
│       │   └── screen_ocr.py
│       │
│       ├── control/
│       │   ├── visual_servo.py
│       │   ├── image_jacobian.py
│       │   ├── press_controller.py
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

The first complete milestone remains intentionally small:

``` text
Input:
"G"
```

Expected behavior:

``` text
Supervisor requests G
        ↓
ACT uses TOP + WRIST + state + target=G
        ↓
ACT creates a servo-ready local viewpoint
        ↓
WRIST perception visually recognizes G
        ↓
ACT queue is stopped/reset
        ↓
wait for follower stability + preserve existing Goal_Position anchor
        ↓
Visual servo moves G toward p_tip
        ↓
alignment stable at current Z level
        ↓
advance one bounded cumulative Z level
        ↓
stop + settle + reobserve WRIST / realign at fixed Z if needed
        ↓
SIDE camera rectification + OCR
        ↓
CONFIRMED_NO_CHANGE ? repeat bounded stage
        ↓
"G" confirmed
        ↓
stop further descent + retract
        ↓
CONFIRMED_SUCCESS
```

Nothing beyond this is required to prove the primary architecture.

Once this primitive is reliable, multi-character typing is repeated
execution plus deterministic recovery.

------------------------------------------------------------------------

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
screen-confirmed success or bounded failure
      ↓
retract
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
result observation        → independent screen vision
recovery planning         → deterministic supervisor
```

The project is intentionally hybrid.
Learning is used where variation and motion priors are valuable.

Classical algorithms are used where geometry, feedback, safety, and task
semantics can be made explicit.

The boundary between them is not hidden: it is part of the system being
studied.
