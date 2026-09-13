# SO101 Closed-Loop Typing

A closed-loop robotic typing system for the SO-101 arm using multi-camera perception, ACT-based motion policies, wrist-camera visual servoing, and screen-based visual self-verification.

## Overview

This project explores how a low-cost SO-101 robotic arm can perform reliable keyboard typing through a combination of:

* visual perception
* imitation learning
* visual servoing
* action chunking
* multi-camera observation
* screen-based result verification
* automatic error recovery

The goal is **not** to build the fastest robotic typist.

The goal is to build a small but complete embodied control system that can:

1. understand which key should be pressed,
2. visually locate the keyboard and target key,
3. move the SO-101 near the target using a learned policy,
4. refine alignment through closed-loop visual servoing,
5. physically press the key,
6. observe the computer screen,
7. determine whether the action succeeded,
8. automatically recover from mistakes,
9. continue until the requested text is correctly typed.

A typical final task is:

```text
TARGET: ROBOT
```

The robot should eventually produce:

```text
ROBOT
```

on the target computer screen using only physical keyboard interaction.

---

# Motivation

The original project explored robotic USB insertion.

That task exposed an important limitation of the SO-101 platform: many manipulation tasks are dominated by mechanical constraints rather than perception or control.

USB connectors and power plugs require reliable grasping, precise orientation, structural rigidity, and insertion force. These factors make it difficult to isolate and study the software topics that are the actual focus of this project.

The current project therefore intentionally avoids demanding grasping and insertion tasks.

Instead, keyboard typing provides a low-force physical interaction task while still requiring a complete perception-action-feedback loop.

The project is specifically intended to study:

```text
Perception
    ↓
State estimation
    ↓
Learned motion policy
    ↓
Visual servo correction
    ↓
Physical interaction
    ↓
Environmental observation
    ↓
Verification
    ↓
Recovery / next action
```

The keyboard is the experimental environment, not the research objective itself.

---

# Core Research Question

The central question is:

> Can a learned motion policy and classical visual servo controller work together to achieve reliable physical interaction, while external visual feedback is used to verify and recover from execution errors?

The project compares different control strategies:

```text
Classical Control
ACT
ACT + Visual Servo
```

with possible future comparison against:

```text
SmolVLA
SmolVLA + Visual Servo
```

The primary architecture is intentionally modular so that the learned policy can later be replaced without redesigning the entire system.

---

# Target System

The intended hardware configuration is:

```text
SO-101 Robot Arm
        │
        ├── Wrist Camera
        │
        ├── Workspace Camera
        │
        └── Screen Camera

External MacBook
        │
        ├── Keyboard
        └── Screen
```

The MacBook is treated as an external physical environment.

The robot should not rely on operating-system keyboard events to determine whether typing succeeded.

Instead, the system should visually observe the screen after each action.

This creates a true external feedback loop:

```text
robot action
    ↓
physical key press
    ↓
computer state changes
    ↓
screen image changes
    ↓
camera observes result
    ↓
system decides next action
```

---

# Camera Roles

The three cameras have different responsibilities.

## Workspace Camera

The workspace camera observes the complete keyboard and the approximate robot workspace.

Its main responsibilities are:

* detect keyboard location,
* estimate keyboard orientation,
* locate the approximate target key,
* provide global spatial context,
* tolerate moderate keyboard translation and rotation.

It is responsible for answering:

> Where should the robot go?

The workspace camera is used mainly for coarse localization.

---

## Wrist Camera

The wrist camera observes the local area around the end effector.

Its main responsibilities are:

* detect the current target key,
* estimate target-key center,
* estimate local alignment error,
* perform fine visual servo correction before pressing.

It is responsible for answering:

> How far am I from the exact target?

The wrist camera is the primary sensor for precision control.

---

## Screen Camera

The screen camera observes a fixed region of the MacBook display.

Its responsibilities are:

* detect the typed-text region,
* recognize currently displayed text,
* verify whether the previous action succeeded,
* provide feedback to the task supervisor.

It answers:

> Did the robot actually do what it intended to do?

The screen camera is deliberately separated from the learned motor policy in the initial system.

It acts as an external verification sensor.

---

# System Architecture

The initial architecture is:

```text
                    Target Text
                     "ROBOT"
                        │
                        ▼
                ┌────────────────┐
                │ Task Supervisor│
                └────────────────┘
                        │
                current target = R
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
     Workspace Perception    Screen Perception
              │                   │
      approximate target       OCR result
              │                   │
              ▼                   │
             ACT                  │
       coarse motion              │
              │                   │
              ▼                   │
        Wrist Camera              │
              │                   │
              ▼                   │
        Visual Servo              │
         fine alignment           │
              │                   │
              ▼                   │
            PRESS                 │
              │                   │
              └──── observe ──────┘
```

The four major control components are intentionally separated.

---

# Responsibility of Each Module

## Task Supervisor

The supervisor manages the high-level task.

Example:

```text
target = "ROBOT"
```

The supervisor decomposes it into:

```text
R
O
B
O
T
```

The learned policy does not need to understand the concept of a word.

It only needs to execute a primitive such as:

```text
PRESS_KEY("R")
```

This separation makes the system easier to understand, test, and debug.

---

## ACT

ACT is responsible for learned coarse motor behavior.

Example input:

```text
workspace image
wrist image
joint state
target key condition
```

Example output:

```text
action chunk
```

The objective of ACT is not necessarily to land perfectly on the center of a key.

Instead, ACT should learn to move the robot into a useful neighborhood around the target.

A possible target condition can be represented as:

```text
A = 0
B = 1
...
Z = 25
SPACE = 26
BACKSPACE = 27
```

The first implementation therefore does not require natural-language conditioning.

---

## Visual Servo

Visual servoing performs local closed-loop correction.

Suppose the wrist camera detects:

```text
target key center = (u_t, v_t)
tool reference    = (u_e, v_e)
```

The image-space error is:

```text
e = [
    u_t - u_e,
    v_t - v_e
]
```

The controller repeatedly performs:

```text
observe
   ↓
measure image error
   ↓
command small motion
   ↓
observe again
```

until the alignment error is below a defined threshold.

Only then should the system press the key.

The initial implementation should use staged control:

```text
ACT
 ↓
Visual Servo
 ↓
PRESS
```

rather than combining both control outputs continuously.

A later experiment may investigate residual control:

```text
u = u_ACT + λ u_VS
```

---

# Screen Verification

After each key press, the robot does not assume that the action succeeded.

Instead:

```text
press key
   ↓
wait for screen update
   ↓
capture screen image
   ↓
OCR
   ↓
compare observed text with target
```

For example:

```text
target:
ROBOT

observed:
ROBO
```

The supervisor determines that the next expected key is:

```text
T
```

and continues.

---

# Automatic Error Recovery

Recovery is automatic, but its semantics are deterministic.

The system must know that:

```text
BACKSPACE
```

removes the previous character.

ACT does not need to understand the linguistic meaning of deletion.

ACT only needs to learn how to physically press the Backspace key when requested.

For example:

```text
target   = ROBOT
observed = ROBOR
```

The longest correct prefix is:

```text
ROBO
```

Therefore the supervisor generates:

```text
PRESS_BACKSPACE
PRESS_T
```

Another example:

```text
target   = ROBOT
observed = ROXX
```

The correct prefix is:

```text
RO
```

The supervisor generates:

```text
PRESS_BACKSPACE
PRESS_BACKSPACE
PRESS_B
PRESS_O
PRESS_T
```

Recovery logic therefore belongs to the task supervisor rather than ACT.

The division of responsibility is:

```text
OCR
"What happened?"

Supervisor
"What should happen next?"

ACT
"How do I approximately perform that action?"

Visual Servo
"How do I execute it accurately?"
```

---

# Initial Supported Keys

The first version should intentionally support only:

```text
A-Z
SPACE
BACKSPACE
```

Future versions may add:

```text
ENTER
SHIFT
numbers
punctuation
```

The initial task length should also remain small.

Recommended first target:

```text
1-5 characters
```

Example tasks:

```text
CAT
DOG
HELLO
ROBOT
VISION
```

---

# Keyboard Setup

The first version should use a fixed MacBook keyboard.

Recommended constraints:

* fixed laptop position,
* fixed screen angle,
* fixed camera mounts,
* fixed keyboard layout,
* predictable lighting,
* large screen font,
* dedicated typing application.

The purpose of V0 is not generalization.

The purpose is to establish a reliable closed-loop system.

Generalization should be introduced only after the basic pipeline works.

---

# Screen Application

A simple dedicated application or webpage should display:

```text
TARGET

ROBOT


TYPED

ROBO
```

The text should initially use:

* large font,
* high contrast,
* fixed position,
* fixed background,
* fixed screen ROI.

The screen perception pipeline can therefore begin with a controlled OCR problem.

Later versions may introduce:

* different font sizes,
* different themes,
* changed window positions,
* variable screen brightness.

---

# End Effector

The project should not depend on grasping.

A simple fixed pressing tool should be attached to the SO-101 gripper.

Possible tools include:

* pen,
* stylus,
* plastic rod,
* rubber-tipped pointer,
* lightweight custom printed tip.

The tool should remain mechanically fixed relative to the wrist camera.

The end-effector reference point should therefore be stable and visually identifiable.

---

# Development Roadmap

## Phase 0 — Mechanical Feasibility

Goal:

> Verify that SO-101 can reliably press a MacBook key using a fixed tool.

No perception.

No ACT.

No visual servo.

Acceptance criteria:

```text
manual / teleoperation key presses
≥ 30 trials
no unstable grasping
no excessive force required
repeatable key activation
```

If this phase fails, software development should stop until the physical setup is corrected.

---

## Phase 1 — Screen Perception

Goal:

> Reliably determine what text appears on the screen.

Pipeline:

```text
screen camera
     ↓
fixed ROI
     ↓
image preprocessing
     ↓
OCR
     ↓
observed string
```

Acceptance criteria:

```text
known test strings
≥ 99% character recognition
under controlled lighting
```

---

## Phase 2 — Workspace Keyboard Detection

Goal:

> Determine the approximate position of a requested key.

Input:

```text
workspace camera image
target key
```

Output:

```text
approximate key location
```

Initial keyboard pose may remain fixed.

Later tests should include:

```text
keyboard translation
keyboard rotation
camera variation
```

---

## Phase 3 — Wrist Visual Servo

Goal:

> Move the tool to the center of a target key using only closed-loop image feedback.

No ACT yet.

Pipeline:

```text
target key detected
        ↓
pixel error
        ↓
small robot motion
        ↓
new image
        ↓
repeat
```

Acceptance criteria should include:

```text
final pixel error
alignment success rate
number of servo iterations
time to convergence
```

---

## Phase 4 — Single-Key Closed Loop

Goal:

> Press one requested key and visually verify the result from the screen.

Example:

```text
target = G

workspace detection
      ↓
coarse motion
      ↓
wrist visual servo
      ↓
press G
      ↓
screen OCR
      ↓
"G"
      ↓
SUCCESS
```

This is the first complete system milestone.

---

## Phase 5 — Multi-Key Typing

Goal:

> Type a short string using repeated execution of the single-key primitive.

Example:

```text
target = CAT
```

Execution:

```text
C
verify
A
verify
T
verify
```

The system should verify every character rather than waiting until the entire word is complete.

---

## Phase 6 — Automatic Error Recovery

Goal:

> Detect mistakes and automatically repair them.

Examples:

```text
TARGET: CAT
TYPED : CAR
```

Recovery:

```text
BACKSPACE
T
```

And:

```text
TARGET: ROBOT
TYPED : ROXX
```

Recovery:

```text
BACKSPACE
BACKSPACE
B
O
T
```

Metrics should include:

```text
error detection accuracy
recovery success rate
average recovery actions
final task success rate
```

---

## Phase 7 — ACT Dataset Collection

After the deterministic closed-loop system works, teleoperation demonstrations can be collected.

Recommended observations:

```text
workspace RGB
wrist RGB
joint positions
joint velocities
target key
```

Recommended action:

```text
SO-101 joint command
```

The screen camera should not initially be included in the ACT observation.

Screen perception belongs to the external supervisor.

---

## Phase 8 — ACT Policy

Train ACT to perform coarse key-approach motion.

The expected behavior is:

```text
arbitrary start configuration
        ↓
ACT
        ↓
move near requested key
```

ACT does not need perfect final accuracy.

Fine alignment remains the responsibility of visual servoing.

---

## Phase 9 — ACT + Visual Servo

Final primary architecture:

```text
ACT
 ↓
coarse target neighborhood
 ↓
switch controller
 ↓
Visual Servo
 ↓
fine alignment
 ↓
PRESS
 ↓
Screen Verification
```

The system should compare:

```text
ACT only
Visual Servo only
ACT + Visual Servo
```

---

# Evaluation

The project should emphasize quantitative evaluation.

## Typing Metrics

```text
character success rate
word success rate
final task success rate
characters per minute
average retries per character
```

---

## Visual Servo Metrics

```text
initial pixel error
final pixel error
servo iterations
convergence time
failure rate
```

---

## Learned Policy Metrics

```text
coarse approach success rate
distance to target after ACT
out-of-distribution success rate
```

---

## Recovery Metrics

```text
error detection accuracy
successful recovery rate
number of corrective actions
final corrected-string accuracy
```

---

# Generalization Experiments

After the base system becomes reliable, introduce controlled perturbations.

## Keyboard Translation

```text
±1 cm
±3 cm
±5 cm
```

## Keyboard Rotation

```text
±2°
±5°
±10°
```

## Robot Initial Configuration

Randomize initial joint positions within a safe region.

## Camera Variation

Introduce small changes in camera position.

## Target Strings

Evaluate strings not present during demonstration collection.

For example:

```text
training demonstrations:
CAT
DOG
HELLO

evaluation:
ROBOT
VISION
OPENAI
```

The important distinction is that evaluation should test composition of known key-press skills rather than memorization of complete word trajectories.

---

# Controller Comparison

One of the primary experiments is:

| System    | Learned Motion | Visual Feedback | External Verification | Recovery |
| --------- | -------------- | --------------- | --------------------- | -------- |
| Classical | No             | Yes             | Yes                   | Yes      |
| ACT       | Yes            | Limited         | Yes                   | Yes      |
| ACT + VS  | Yes            | Yes             | Yes                   | Yes      |

Possible future extension:

| System       | Language Conditioned |
| ------------ | -------------------- |
| ACT          | No / symbolic target |
| SmolVLA      | Yes                  |
| SmolVLA + VS | Yes                  |

This allows the project to investigate whether learned action policies and classical feedback controllers provide complementary capabilities.

---

# Why ACT First?

The initial project uses ACT rather than a VLA because the high-level language problem is simple and deterministic.

For example:

```text
"ROBOT"
```

can be decomposed programmatically into:

```text
R
O
B
O
T
```

There is little benefit in asking a large vision-language-action model to discover this decomposition.

The difficult part is physical execution.

ACT therefore provides a simpler platform for studying:

* demonstration collection,
* action chunking,
* learned motion priors,
* visual generalization,
* hybrid learned/classical control.

A VLA such as SmolVLA can later be introduced as an additional experiment rather than a dependency of the core system.

---

# Non-Goals

The initial project does **not** attempt to solve:

* high-speed robotic typing,
* arbitrary keyboards,
* arbitrary laptop models,
* arbitrary camera placement,
* unrestricted natural-language instruction following,
* dexterous grasping,
* force-controlled manipulation,
* human-level typing speed,
* end-to-end VLA control.

These may become future experiments.

The first objective is reliability and interpretability.

---

# Safety

The SO-101 should operate at conservative speed and acceleration limits.

The pressing tool must not contain sharp or conductive surfaces that could damage the laptop.

The system should define:

```text
joint limits
workspace limits
maximum downward motion
maximum press duration
timeout conditions
emergency stop behavior
```

A failed visual detector must never result in uncontrolled downward motion.

All press operations should be bounded by predefined safe motion limits.

---

# Proposed Repository Structure

```text
so101_closed_loop_typing/
│
├── configs/
│   ├── cameras/
│   ├── robot/
│   ├── visual_servo/
│   └── act/
│
├── src/
│   └── so101_typing/
│       ├── adapters/
│       │   ├── robot.py
│       │   └── cameras.py
│       │
│       ├── perception/
│       │   ├── keyboard.py
│       │   ├── key_detector.py
│       │   ├── tool_detector.py
│       │   └── screen_ocr.py
│       │
│       ├── control/
│       │   ├── coarse_control.py
│       │   ├── visual_servo.py
│       │   ├── press_controller.py
│       │   └── safety.py
│       │
│       ├── policy/
│       │   ├── act_policy.py
│       │   └── dataset.py
│       │
│       ├── supervisor/
│       │   ├── typing.py
│       │   ├── verification.py
│       │   └── recovery.py
│       │
│       ├── runtime/
│       │   ├── single_key.py
│       │   ├── typing.py
│       │   └── evaluate.py
│       │
│       └── utils/
│
├── tests/
│
├── scripts/
│   ├── calibrate_cameras.py
│   ├── test_screen_ocr.py
│   ├── test_visual_servo.py
│   ├── collect_act_data.py
│   ├── train_act.py
│   └── run_typing_demo.py
│
├── docs/
│   ├── architecture.md
│   ├── calibration.md
│   ├── dataset.md
│   └── experiments.md
│
└── README.md
```

---

# First End-to-End Milestone

The first complete milestone is intentionally small:

```text
Input:
"G"
```

Expected behavior:

```text
Workspace camera detects keyboard
            ↓
System determines approximate G location
            ↓
SO-101 moves near G
            ↓
Wrist camera detects G
            ↓
Visual servo centers the tool
            ↓
SO-101 presses G
            ↓
Screen camera observes the display
            ↓
OCR returns "G"
            ↓
SUCCESS
```

Nothing beyond this milestone should be considered necessary for proving the core architecture.

Once this works reliably, multi-character typing becomes repeated execution of the same primitive.

---

# Final Demonstration

A target such as:

```text
TARGET: ROBOT
```

is displayed or provided to the controller.

The SO-101 physically types:

```text
R
O
B
O
T
```

while verifying the screen after every action.

If the robot accidentally produces:

```text
ROBOR
```

the system detects the mismatch and automatically performs:

```text
BACKSPACE
T
```

until the screen shows:

```text
ROBOT
```

The final objective is therefore not simply:

> make the robot press keys

but:

> build a robot that perceives, acts, observes the consequence of its action, detects mistakes, and autonomously corrects them.

---

# Long-Term Extensions

Possible extensions include:

* SmolVLA as a language-conditioned policy,
* natural-language commands such as `type robot`,
* shifted keyboard layouts,
* multiple keyboard models,
* punctuation and modifier keys,
* continuous ACT + visual-servo residual control,
* learned recovery policies,
* uncertainty-aware OCR,
* multi-camera policy fusion,
* online replanning,
* autonomous calibration,
* transfer to other button-based interaction tasks.

The architecture should remain modular enough that these extensions do not require redesigning the core runtime.

---

# Project Principle

The project follows one central engineering rule:

> Solve deterministic problems deterministically, and use learning where learning provides real value.

Therefore:

```text
string comparison      → deterministic code
typing progress        → task supervisor
screen recognition     → vision / OCR
coarse robot motion    → ACT
fine alignment         → visual servo
physical execution     → SO-101
result verification    → external visual feedback
```

This separation is the foundation of the project.
