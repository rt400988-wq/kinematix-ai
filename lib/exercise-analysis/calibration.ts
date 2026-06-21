// The pre-exercise "Calibration" phase: frame check -> ready-position check
// -> 3-2-1 countdown -> done. Owned separately from ExerciseAnalyzer (which
// only runs once calibration is done) so the two state machines don't
// tangle — see analyzer.ts for where control hands off.

import type { ExerciseMode, Landmark } from "./types"
import { NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE } from "./types"
import { angleAt, lineAngleFromHorizontal, midpoint, visible } from "./geometry"
import { PersistentFlagSeconds } from "./smoothing"

export type CalibrationStepId = "frame" | "position" | "countdown" | "done"

export type CalibrationStatus = {
  step: CalibrationStepId
  message: string
  /** Seconds remaining, only meaningful during the "countdown" step. */
  countdownValue?: number
}

// How long a person must continuously satisfy a calibration step before
// advancing — same noise-rejection reasoning as PersistentFlag, just
// expressed as seconds since these checks are coarser/slower than per-rep
// form alerts.
const FRAME_CHECK_SECONDS = 1.0
const POSITION_CHECK_SECONDS = 1.0
const COUNTDOWN_SECONDS = 3

type ReadyPositionCheck = (lm: Landmark[]) => boolean

function isStandingUpright(lm: Landmark[]): boolean {
  const lk = lm[L_KNEE], rk = lm[R_KNEE], lh = lm[L_HIP], rh = lm[R_HIP], la = lm[L_ANKLE], ra = lm[R_ANKLE]
  if (![lk, rk, lh, rh, la, ra].every(visible)) return false
  // "Standing" ~= knees are close to straight (large hip-knee-ankle angle)
  // on at least one visible side.
  const leftAngle = angleAt(lh, lk, la)
  const rightAngle = angleAt(rh, rk, ra)
  return Math.max(leftAngle, rightAngle) > 150
}

function isInPlankLikePosition(lm: Landmark[]): boolean {
  const ls = lm[L_SHOULDER], rs = lm[R_SHOULDER], lh = lm[L_HIP], rh = lm[R_HIP]
  if (![ls, rs, lh, rh].every(visible)) return false
  // Roughly horizontal torso (shoulder-midpoint to hip-midpoint line close
  // to flat) is a cheap proxy for "in a plank/push-up start position"
  // without requiring ankle visibility, since feet are often out of frame
  // on tighter shots.
  const shoulderMid = midpoint(ls, rs)
  const hipMid = midpoint(lh, rh)
  const torsoFlatness = lineAngleFromHorizontal(shoulderMid, hipMid)
  return torsoFlatness < 35
}

function isGuardUpBothHands(lm: Landmark[]): boolean {
  const lw = lm[L_WRIST], rw = lm[R_WRIST], nose = lm[NOSE]
  if (![lw, rw, nose].every(visible)) return false
  // Both wrists at or above chin height (nose.y is a close, always-visible proxy for chin height).
  return lw.y <= nose.y + 0.1 && rw.y <= nose.y + 0.1
}

// Mode-specific "are you in the starting position" checks. Deliberately
// loose (intermediate, not strict) — these only gate the countdown, not
// rep scoring, so erring toward permissive here just means calibration
// finishes a bit faster, not that bad reps get scored as good ones.
const READY_CHECKS: Record<ExerciseMode, ReadyPositionCheck> = {
  Squats: (lm) => isStandingUpright(lm),
  Lunges: (lm) => isStandingUpright(lm),
  "Push-ups": (lm) => isInPlankLikePosition(lm),
  Plank: (lm) => isInPlankLikePosition(lm),
  Shadowboxing: (lm) => isGuardUpBothHands(lm),
  "Jab-Cross": (lm) => isGuardUpBothHands(lm),
}

function isFullBodyInFrame(lm: Landmark[], mode: ExerciseMode): boolean {
  // What "full body" means depends on the exercise — match the limb chain
  // each mode's tracking logic actually depends on, rather than requiring
  // every landmark regardless of relevance:
  //   - Squats/Lunges need legs (that's what's measured).
  //   - Plank/Push-ups need one full arm chain, shot more tightly/from the side.
  //   - Shadowboxing/Jab-Cross need wrists — requiring ankle visibility here
  //     would fail calibration for a perfectly normal waist-up framing.
  const core = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP].map((i) => lm[i])
  if (!core.every(visible)) return false

  if (mode === "Plank" || mode === "Push-ups") {
    return [lm[L_ELBOW], lm[L_WRIST]].every(visible) || [lm[R_ELBOW], lm[R_WRIST]].every(visible)
  }
  if (mode === "Shadowboxing" || mode === "Jab-Cross") {
    return visible(lm[L_WRIST]) && visible(lm[R_WRIST])
  }
  const oneLegChain = [lm[L_KNEE], lm[L_ANKLE]].every(visible) || [lm[R_KNEE], lm[R_ANKLE]].every(visible)
  return oneLegChain
}

function positionMessage(mode: ExerciseMode): string {
  switch (mode) {
    case "Squats":
    case "Lunges":
      return "Stand upright with your feet shoulder-width apart."
    case "Push-ups":
    case "Plank":
      return "Get into a straight-arm plank position, filmed from the side if possible."
    case "Shadowboxing":
    case "Jab-Cross":
      return "Bring both hands up to guard your chin."
    default: {
      const _exhaustive: never = mode
      return _exhaustive
    }
  }
}

/**
 * Drives the "Calibration" pre-exercise phase: frame check -> ready-position
 * check -> 3-2-1 countdown -> done.
 */
export class CalibrationController {
  private step: CalibrationStepId = "frame"
  private frameOkFlag = new PersistentFlagSeconds(FRAME_CHECK_SECONDS)
  private positionOkFlag = new PersistentFlagSeconds(POSITION_CHECK_SECONDS)
  private countdownStartedAt: number | null = null

  reset() {
    this.step = "frame"
    this.frameOkFlag.reset()
    this.positionOkFlag.reset()
    this.countdownStartedAt = null
  }

  get currentStep() {
    return this.step
  }

  /**
   * Advances the calibration state machine by one frame and returns the
   * current step + message to show the user. When this returns
   * `step: "done"` for the first time, the caller should immediately call
   * `ExerciseAnalyzer.captureBaseline(landmarks, mode)` with that same
   * frame's landmarks, then start feeding frames to `ExerciseAnalyzer`
   * instead of this controller.
   */
  process(landmarks: Landmark[] | undefined, mode: ExerciseMode, timestampMs: number): CalibrationStatus {
    if (this.step === "done") return { step: "done", message: "" }

    if (!landmarks || landmarks.length < 33) {
      // Lost tracking entirely — fall back to the frame-check message but
      // don't reset progress on a single dropped frame.
      return { step: this.step === "frame" ? "frame" : this.step, message: "Make sure your camera can see you." }
    }

    if (this.step === "frame") {
      const ok = this.frameOkFlag.update(isFullBodyInFrame(landmarks, mode), timestampMs)
      if (ok) {
        this.step = "position"
      } else {
        return { step: "frame", message: "Step back so your full body is in frame." }
      }
    }

    if (this.step === "position") {
      const check = READY_CHECKS[mode]
      const ok = this.positionOkFlag.update(check(landmarks), timestampMs)
      if (ok) {
        this.step = "countdown"
        this.countdownStartedAt = timestampMs
      } else {
        return { step: "position", message: positionMessage(mode) }
      }
    }

    if (this.step === "countdown") {
      const elapsed = (timestampMs - (this.countdownStartedAt ?? timestampMs)) / 1000
      const remaining = Math.max(0, Math.ceil(COUNTDOWN_SECONDS - elapsed))
      if (remaining <= 0) {
        this.step = "done"
        return { step: "done", message: "Go!" }
      }
      return { step: "countdown", message: `Perfect, starting in ${remaining}...`, countdownValue: remaining }
    }

    return { step: "done", message: "" }
  }
}
