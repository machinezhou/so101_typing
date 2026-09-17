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
7.  once alignment is stable, descend in bounded pure-Z steps, stopping after
    every step to reacquire vision and realign XY if needed,
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

The project is no longer at the mechanical-feasibility stage.

The current physical setup has already demonstrated:

- a calibrated SO-101 follower/leader setup,

- stable teleoperation,

- a previous ACT block-manipulation demo,

- a pencil mechanically attached to the gripper with cable ties,

- successful physical MacBook key presses using that tool,

- three simultaneous camera streams,

- a fixed MacBook and robot workspace.
  Therefore, the main engineering risk has shifted away from *“can the
  SO-101 physically press a key?”* toward:

- local visual recognition of key identity,

- robust ACT-to-visual-servo handoff,

- image-space closed-loop alignment,

- screen verification,

- runtime ownership and failure handling,

- reproducible data collection and evaluation.

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
              │                               │
              ▼                               │
       VISUAL SERVO (XY)                      │
   target center → pencil-tip pixel           │
              │                               │
        alignment stable?                     │
              │ yes                           │
              ▼                               │
       PURE-Z DESCENT STEP                    │
              │                               │
        stop + settle                         │
              │                               │
       fresh WRIST observation                │
        ├─ misaligned → XY realign             │
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

The important architectural property is **controller ownership**:

- ACT owns coarse motion only until a perception-ready handoff.
- After handoff, the deterministic staged controller owns XY alignment and
  bounded pure-Z descent; ACT does not resume during the press attempt.
- Screen perception never commands the robot directly, but its confirmed
  result determines whether further descent is allowed.
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
  ├── refresh joint state
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
  pure-Z descent step, not continuous downward motion.

Every Z step invalidates the previous alignment acceptance. The controller must
stop, settle, acquire fresh observations, and realign if necessary. If
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

The handoff must be triggered by perception, not by a fixed time or a
fixed number of ACT steps.

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

When `servo_ready` becomes true:

1.  stop ACT,
2.  clear/reset any pending ACT action chunk,
3.  prevent any stale ACT action from reaching the robot,
4.  read the latest robot state,
5.  wait for/obtain a fresh wrist frame,
6.  transfer exclusive control ownership to the deterministic staged
    visual-servo/press controller.
    This boundary is critical. ACT and the deterministic controller must never
    command the robot concurrently in V0. Handoff does not require the target
    to already be precisely aligned with the pencil-tip reference.

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

After ACT/manual handoff, WRIST vision owns fine alignment. The V0 staged press
uses the same alignment rule at every stopped Z level:

``` text
XY ALIGNMENT
    ↓
ALIGNED_AT_LEVEL
    ↓
ONE PURE-Z STEP
    ↓
STOP + SETTLE + REOBSERVE
    ↓
REALIGN XY IF NEEDED
```

Do not command lateral correction and downward motion simultaneously.

## Tool-tip reference point

The control reference is the physical pencil-tip projection in the rigid WRIST
camera/tool image:

``` text
p_tip = (u_tip, v_tip)
```

`p_tip` must be calibrated/validated directly as a WRIST tool property. The old
H3.1 value obtained by manually placing the pencil over `G` must not be silently
relabelled as a direct pencil-tip calibration.

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

## Local image Jacobian

For V0, the local mapping from a small **requested Cartesian XY command**
to image motion can be estimated experimentally. The millimetre value is
the command-space scaling used by the LeRobot Cartesian processor; it is
not an independently measured TCP displacement or an SO-101 positioning-
accuracy claim.

Apply small safe perturbations around a representative pre-press pose:

``` text
+x
-x
+y
-y
```

and measure the corresponding target-center displacement in the wrist
image.

Estimate the local command-space image Jacobian:

``` text
delta_p_image ~= J_cmd @ delta_c_requested
```

where `J_cmd` has units `px / commanded-mm`.

H3.2 has already accepted this command-space mapping. The existing canonical
`calibration/image_jacobian.json` remains the source of truth. Replacing the
old target-derived H3.1 reference with a directly calibrated `p_tip` does not
by itself invalidate H3.2.

Then a local controller can use a damped/bounded form of:

``` text
delta_c_requested = -J_cmd^+ @ e
```

with:

- bounded Cartesian XY step size,
- damping if necessary,
- maximum iteration/correction budget,
- convergence threshold,
- target-loss handling,
- fresh-frame requirement.

The Cartesian correction can then be converted to safe robot commands through
the existing LeRobot kinematic/IK path.

## Alignment acceptance

Do not declare alignment from one frame.

A robust rule should require:

``` text
||p_key - p_tip|| < epsilon
for N consecutive fresh frames
```

before authorizing one downward step. After every Z step, this acceptance is
invalidated and must be established again from fresh WRIST observations.

For V0, multi-frame acceptance is the first anti-chatter mechanism. If hardware
logs later show threshold chatter or small left/right oscillation near
convergence, add an alignment deadband/hysteresis buffer (and, if needed,
temporal filtering or reduced near-target gain) without changing the staged
control architecture. Do not choose the buffer width until real hardware data
shows the noise/oscillation scale.

------------------------------------------------------------------------

# Deterministic Press Controller

The press stage should not be learned in V0. It is a bounded staged descent,
not one precomputed downward stroke:

``` text
stable XY alignment at current level
        ↓
one bounded pure-Z step
        ↓
stop + settle
        ↓
fresh WRIST observation
        ↓
realign XY if needed
        ↓
independent SIDE/screen verification
        ↓
CONFIRMED_SUCCESS ? retract : next bounded level
```

The controller must define:

- per-step downward displacement,
- maximum cumulative downward displacement,
- speed/acceleration limits,
- settle/fresh-frame requirements,
- workspace bounds,
- timeout behavior,
- abort/retract behavior.

A completed motion command does **not** mean the arm is physically settled. The
SO-101 may show a short post-motion mechanical wobble, so observations captured
during the settling window must not authorize a correction or another Z step.
V0 may use a conservative settle delay plus consecutive stable fresh frames; a
later optimization may replace the fixed delay with measured image/joint
stability when hardware logs justify it.

The pencil/tool must never move downward if target confidence or current-level
alignment is invalid. Commanded descent depth is a safety/budget quantity, not
a success detector. Further descent stops immediately when screen verification
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
2.260  z_step=-0.5mm cumulative_z=-0.5mm
2.420  transition=SETTLE_AND_REOBSERVE->VERIFY_LEVEL
2.610  verification=CONFIRMED_NO_CHANGE
2.611  transition=VERIFY_LEVEL->DESCEND_STEP
2.670  z_step=-0.5mm cumulative_z=-1.0mm
2.840  alignment_error=(+2,+1)
2.980  screen_text="R"
2.981  verification=CONFIRMED_SUCCESS
2.982  transition=VERIFY_LEVEL->RETRACT
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
the project. The detailed technical definition and acceptance criteria
for each phase remain unchanged in the Development Roadmap below.

## Overall Progress

| Phase    | Description                                         | Status                    |
|----------|-----------------------------------------------------|---------------------------|
| Phase 0  | Mechanical Feasibility                              | **Completed**             |
| Phase 1  | Freeze Camera Geometry and Build Camera Sanity Tool | **Completed**             |
| Phase 2  | Wrist Keycap Detection and Glyph Recognition        | **Completed**             |
| Phase 3  | Tool Reference and Visual Servo                     | **IN PROGRESS — CURRENT** |
| Phase 4  | Screen Rectification and Verification               | Not Started               |
| Phase 5  | Deterministic Local Single-Key Closed Loop          | Not Started               |
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
Phase 3 — Tool Reference and Visual Servo
```

Current checkpoint:

``` text
Phase 1 camera/software foundation
        ↓
PHASE 1 ACCEPTED / FROZEN
        ↓
Phase 2 wrist perception
        ↓
PHASE 2 ACCEPTED / FROZEN
        ↓
PHASE 3 TOOL REFERENCE + VISUAL SERVO     <-- CURRENT
        ↓
legacy H3.1 target-derived p*              ✓ historical checkpoint
        ↓
LeRobot official Cartesian backend        ✓
        ↓
fixed-anchor Cartesian planning            ✓
        ↓
H3.2 conditioned physical calibration      ✓ ACCEPTED
        ↓
conditioning convergence                   ✓ 3 cycles
        ↓
formal paired samples                      ✓ 8 samples
        ↓
J_cmd [px/commanded-mm]                    ✓
[[-2.386845, -0.414119],
 [ 0.294672,  2.875000]]
        ↓
residual RMS / condition number            ✓ 3.256 px / 1.391
        ↓
direction opposition X / Y                 ✓ -0.981 / -0.931
        ↓
canonical image_jacobian.json              ✓ PROMOTED
        ↓
direct WRIST pencil-tip reference p_tip   <-- NEXT
        ↓
closed-loop p_key -> p_tip XY validation
        ↓
small-budget staged Z / reobserve / realign validation
        ↓
Phase 3 acceptance
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
PHASE 3 IN PROGRESS — CURRENT
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
PHASE 3 IN PROGRESS — CURRENT
```

## Phase 3 Current Checkpoint

Phase 3 starts from the accepted wrist-perception runtime and does not
require ACT. The current task is to validate the deterministic WRIST-controlled
primitive from a safe teleoperated pose: direct pencil-tip reference,
`p_key -> p_tip` XY alignment, then a small-budget staged descent that stops and
reobserves after every Z step. Full screen-confirmed keypress completion remains
Phase 5 integration.

Completed or hardware-validated at this checkpoint:

- H3.1 remains a historical calibration checkpoint: it measured the target-key
  center when the operator believed the pencil was aligned over `G`. It must
  not be silently reinterpreted as a direct physical pencil-tip pixel
  calibration. The new staged-control design therefore requires a direct
  WRIST `p_tip` calibration/validation before hardware staged-servo execution.
- Cartesian motion delegates FK, end-effector bounds/safety processing, IK,
  and joint-target generation to LeRobot's SO-101 Cartesian processor path
  instead of maintaining a project-local IK contract.
- Project-facing visual-servo corrections use `base_link_xy` and
  **commanded millimetres**. These are requested Cartesian command units,
  not independently measured TCP displacement or positioning-accuracy
  claims. LeRobot owns conversion through the robot kinematics path.
- The project-side Cartesian regression investigated before physical H3.2 was
  isolated and corrected. A same-target hardware hold repeatedly sent one
  unchanged joint target at 30 Hz and passed with **0.000 deg observed span
  on every joint**. The corrected fixed-anchor zero request also reproduced
  the anchor with **0.000 deg maximum planned joint shift**.
- A fixed-anchor `+X 5.00` commanded-mm plan check returned approximately
  `(+4.95,-0.01,-0.09) mm` relative to the model anchor, with **0.05 mm XY
  model error** and **0.09 mm Z model error**. This remains a model-consistency
  and safety check, not a physical TCP-accuracy claim.
- H3.2 now uses an **adaptive conditioned fixed-anchor protocol**. Conditioning
  runs repeated `+X/-X/+Y/-Y` cycles as readiness-only data and never includes
  those samples in the Jacobian fit. Readiness requires adjacent same-phase
  cycles to satisfy the configured image-position drift, image-response drift,
  and X/Y opposition gates.
- On the accepted physical run, conditioning correctly rejected the first
  adjacent-cycle comparison and converged on the next one. The protocol passed
  after **3 conditioning cycles**. Only then did the script start a fresh
  formal H3.2 dataset.
- The formal dataset contains **8** paired `+X/-X/+Y/-Y` motion samples. The
  accepted command-space image Jacobian is:

  ``` text
  J_cmd [px/commanded-mm] =
  [[-2.386845397949219,   -0.41411895751953126],
   [ 0.29467163085937503,  2.87500000000000000]]
  ```

- Formal H3.2 acceptance metrics were **residual RMS = 3.256 px**,
  **condition number = 1.391**, **X opposition = -0.981**, and
  **Y opposition = -0.931**. The candidate was accepted with no reported
  acceptance errors.
- The accepted candidate was explicitly promoted with
  `scripts/promote_image_jacobian.py --confirm-reviewed`. The canonical
  calibration is now `calibration/image_jacobian.json` with
  `sample_count = 8` and `input_semantics = requested_cartesian_delta`.
- Measured-joint FK remains **diagnostic-only** for H3.2. The calibrated
  mapping is `requested Cartesian XY delta -> observed WRIST pixel delta`;
  measured FK must not be reintroduced as an open-loop millimetre-accuracy
  acceptance gate.
- The pre-H3.2 software baseline had **141/141 unit tests passing**. The final
  promotion checkpoint also passed Python compilation and `git diff --check`.
- **H3.2 is complete and should now be treated as a saved calibration
  checkpoint.** Do not repeat it unless the fixed camera/tool/keyboard geometry
  changes or later closed-loop evidence contradicts the local calibration.

The remaining Phase 3 sequence is:

``` text
canonical J_cmd available                    ✓
        ↓
directly calibrate / validate WRIST p_tip     <-- NEXT
        ↓
run bounded p_key -> p_tip XY servo on hardware
        ↓
require stable multi-frame alignment
        ↓
authorize one small pure-Z step
        ↓
stop + settle + fresh WRIST observation
        ↓
realign XY if needed
        ↓
repeat only within a small cumulative Z budget
        ↓
measure alignment / re-alignment / target-loss / safety behavior
        ↓
Phase 3 acceptance
```

### Phase 3 implementation notes / pitfalls

These notes capture lessons from the current integration work so later
iterations do not reopen already-resolved branches:

- Do **not** treat FK -> IK -> FK consistency inside one URDF model as proof of
  real-world millimetre accuracy. It validates model consistency, not the
  physical SO-101.
- Do **not** make sub-millimetre open-loop Cartesian accuracy a prerequisite
  for Phase 3. The low-cost arm has mechanical backlash/compliance and the
  project already has visual feedback; fine alignment should be closed-loop
  and measured in the image.
- Prefer LeRobot's existing SO-101 kinematics, calibration, and Cartesian
  processors over duplicating servo calibration or building another IK
  wrapper. The follower calibration remains owned by LeRobot.
- Do **not** treat the rejected V3 current-state incremental rebasing behavior
  as an accepted runtime contract. H3.2 was accepted under conditioned
  fixed-anchor command semantics. Runtime XY integration should preserve those
  validated local semantics; if a larger residual error requires a new local
  anchor/segment, the new segment must pass an explicit image-response/readiness
  check before the canonical `J_cmd` is trusted again.
- The calibrated Jacobian is explicitly a **command-space** mapping:
  `requested Cartesian XY delta -> observed WRIST pixel delta`. FK-derived
  displacement may be logged for diagnostics, but it is not a Phase 3
  positioning-accuracy metric.
- Keep safety limits distinct from accuracy requirements. A maximum Cartesian
  step is a motion bound, not a claim that the arm can position to that
  tolerance.
- Keep the adaptive H3.2 conditioning protocol. The accepted run demonstrated
  a real first-cycle transient; conditioning prevented that transient from
  contaminating the formal Jacobian fit. Conditioning samples are readiness
  data only and must remain excluded from `J_cmd` fitting.
- Do not relatch the fixed Cartesian command anchor during conditioning. The
  accepted protocol keeps one fixed command anchor, detects convergence from
  adjacent complete cycles, and starts a fresh formal dataset only after
  readiness passes.
- Use the SO-101 URDF from LeRobot's configured cache (`HF_LEROBOT_HOME`);
  do not introduce a second project-specific URDF location without a concrete
  reason.
- **Observed WRIST occlusion constraint:** when the pencil tip is less than
  roughly **1 cm above the keyboard**, the pencil/tool can occlude the target
  key/glyph. During staged descent, stop after every Z step, treat degraded
  visibility as target loss, and never combine lateral correction with the
  downward command itself.
- Robust burst consensus remains a defensive guard against transient occlusion
  or a wrong glyph candidate entering one measurement burst.
- Do not reopen the old same-target drift / fixed-anchor planning investigation
  unless a later physical measurement produces contradictory evidence. The
  dedicated hold test and corrected fixed-anchor planning checks already closed
  that branch.
- The next hardware gates are **direct `p_tip` calibration/validation**, then
  closed-loop `p_key -> p_tip` XY validation, followed by a deliberately small
  cumulative-Z staged-descent test. H3.2 is not reopened unless later evidence
  contradicts it.

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
press ENTER to freeze the manual pose
        ↓
autonomous bounded XY visual-servo corrections
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

- **normal Phase 3 completion:** stop autonomous corrections and resume
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
latch one fixed Cartesian command anchor
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

ACT and full screen-confirmed keypress completion remain outside the current
Phase 3 checkpoint. Phase 3 may exercise only a deliberately small, bounded
staged descent to validate stop/reobserve/realign behavior before Phase 5.

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

## Phase 3 — Tool Reference and Visual Servo

Goal:

> Starting from a safe teleoperated local pose, align the detected target key
> to a directly calibrated WRIST pencil-tip reference and validate the staged
> stop/reobserve/realign primitive with a small Z budget.

Tasks:

1.  directly calibrate/validate `p_tip` as the physical pencil-tip projection
    in the WRIST image; do not silently reuse the old target-derived H3.1 `p*`,
2.  retain the already accepted canonical H3.2 `J_cmd`,
3.  integrate bounded `p_key -> p_tip` XY correction with the SO-101 hardware
    executor,
4.  validate fresh-frame, target-loss, FOV/safety, timeout, and correction-budget
    handling,
5.  require stable multi-frame alignment before any Z motion,
6.  validate a small-budget staged loop: one pure-Z step -> stop/settle -> fresh
    WRIST observation -> XY realignment if needed,
7.  repeat `G` alignment from several initial image offsets and perform two
    cross-keyboard transfer sanity checks without per-key references or
    per-key Jacobians,
8.  measure final alignment error, convergence time, iterations, target-loss
    rate, re-alignment behavior, and safety-bound behavior.

Acceptance should include:

- reliable `p_key -> p_tip` convergence from local offsets,
- convergence rate, final pixel error, convergence time, and servo iterations,
- stable alignment before each permitted descent step,
- no simultaneous XY+Z command,
- fresh observation after every Z step,
- successful re-alignment when descent introduces image error,
- target-loss / timeout / failure rate,
- hard stop/recovery on target loss, stale frames, timeout, or cumulative-Z
  budget exhaustion.

No ACT is required for this phase. Full keypress success detection is not a
Phase 3 acceptance requirement.

------------------------------------------------------------------------

## Phase 4 — Screen Rectification and Verification

Goal:

> Reliably determine whether a stopped physical press level produced the
> expected screen change.

Pipeline:

``` text
SIDE
  ↓
screen homography
  ↓
canonical screen
  ↓
fixed typing ROI
  ↓
OCR / text recognition
  ↓
SUCCESS / NO_CHANGE / WRONG / UNCERTAIN
```

Acceptance:

- controlled test strings,
- stable perspective rectification,
- high character recognition accuracy,
- low false-SUCCESS / false-WRONG rate,
- reliable distinction between confirmed no-change and uncertain OCR,
- explicit uncertainty behavior.

------------------------------------------------------------------------

## Phase 5 — Deterministic Local Single-Key Closed Loop

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
one pure-Z step
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
maximum downward motion per staged Z step
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
5.  XY correction and Z descent are never commanded simultaneously,
6.  ACT and the deterministic staged controller never command simultaneously,
7.  failed/uncertain verification never causes an unbounded descent or retry loop.

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
│   ├── test_visual_servo.py
│   ├── test_screen_verification.py
│   ├── collect_act_data.py
│   ├── train_act.py
│   └── run_typing_demo.py
│
├── tests/
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
Visual servo moves G toward p_tip
        ↓
alignment stable at current Z level
        ↓
one bounded pure-Z step
        ↓
stop + settle + reobserve WRIST / realign if needed
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
p_key -> p_tip visual alignment
      ↓
bounded staged Z / stop / reobserve / realign
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
