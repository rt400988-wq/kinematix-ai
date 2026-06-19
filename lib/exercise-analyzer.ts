// Pure exercise-analysis logic, decoupled from React so it's easy to
// unit-test and easy to retune thresholds later. Takes MediaPipe's own
// NormalizedLandmark type directly at the public boundary (rather than a
// parallel custom shape) so there's no structural-compatibility guesswork
// at the call site in app/page.tsx.
//
// Two genuinely different detection strategies live here:
//
//   1. Squats / Lunges — an established technique (angle at hip-knee-ankle,
//      state machine on threshold crossing). See e.g.
//      https://learnopencv.com/ai-fitness-trainer-using-mediapipe/
//      This is real biomechanics: a squat has a clean down/up cycle.
//
//   2. Shadowboxing / Jab-Cross — punches have no equivalent literature
//      precedent. There's no clean "down/up" position pair the way a squat
//      has — strikes are fast, ballistic, and asymmetric. What's implemented
//      here is a velocity-spike heuristic (wrist extension speed crossing a
//      threshold and returning), which is a reasonable approximation but a
//      meaningfully weaker signal than the angle-based squat logic. This is
//      why the UI calls these "Strikes Detected" rather than "Reps" —
//      labeling the precision difference rather than hiding it.

import type { NormalizedLandmark } from "@mediapipe/tasks-vision"

export type Landmark = NormalizedLandmark
export type Tone = "good" | "warn" | "info"
export type ExerciseMode = "Shadowboxing" | "Squats" | "Lunges" | "Jab-Cross"

export type FrameResult = {
  repCompleted: boolean
  formScore: number | null // 0-100, this rep's form quality, null if no rep judged yet
  cue: { text: string; tone: Tone } | null
  trackingOk: boolean // false if required landmarks aren't visible enough to analyze
}

// MediaPipe BlazePose landmark indices.
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const L_SHOULDER = 11, R_SHOULDER = 12
const L_ELBOW = 13, R_ELBOW = 14
const L_WRIST = 15, R_WRIST = 16
const L_HIP = 23, R_HIP = 24
const L_KNEE = 25, R_KNEE = 26
const L_ANKLE = 27, R_ANKLE = 28

const VISIBILITY_MIN = 0.5 // "intermediate" tolerance — strict would be ~0.7+

function angleAt(a: Landmark, vertex: Landmark, b: Landmark): number {
  // Angle at `vertex`, formed by rays to `a` and `b`, in degrees.
  const v1x = a.x - vertex.x, v1y = a.y - vertex.y
  const v2x = b.x - vertex.x, v2y = b.y - vertex.y
  const dot = v1x * v2x + v1y * v2y
  const mag1 = Math.hypot(v1x, v1y)
  const mag2 = Math.hypot(v2x, v2y)
  if (mag1 === 0 || mag2 === 0) return 180
  const cos = Math.min(1, Math.max(-1, dot / (mag1 * mag2)))
  return (Math.acos(cos) * 180) / Math.PI
}

function visible(lm: Landmark | undefined): lm is Landmark {
  return !!lm && (lm.visibility ?? 1) >= VISIBILITY_MIN
}

// Small moving average so a single noisy frame can't flip a state machine
// and produce a phantom rep. Window of 5 at ~30fps is ~165ms — enough to
// smooth jitter without meaningfully lagging real movement.
class RollingAverage {
  private buf: number[] = []
  constructor(private size = 5) {}
  push(v: number): number {
    this.buf.push(v)
    if (this.buf.length > this.size) this.buf.shift()
    return this.buf.reduce((a, b) => a + b, 0) / this.buf.length
  }
  reset() {
    this.buf = []
  }
}

type SquatLegState = {
  phase: "up" | "down"
  angleAvg: RollingAverage
  minAngleThisRep: number
  // NOTE: an earlier version of this scored torso lean against a fixed
  // "ideal" angle, but there's no single correct torso-lean reference —
  // proper forward lean varies legitimately with femur length, stance
  // width, and squat style (can be as low as ~55° at parallel depth for
  // some lifters). Scoring against a fixed constant would have penalized
  // correct form for plenty of body types, so depth is the only component
  // scored here. If you want torso-lean feedback later, it needs a
  // per-person calibration step, not a global constant.
}

type WristState = {
  prevPos: { x: number; y: number } | null
  prevT: number | null
  speedAvg: RollingAverage
  armed: boolean // true once speed has crossed the "extending" threshold, waiting for retraction
  peakSpeed: number
}

// Angle thresholds tuned for the "intermediate / balanced" setting the
// person asked for — forgiving enough that imperfect tablet lighting or a
// slightly-off camera angle won't silently fail to count reps. A "strict"
// mode would tighten DOWN_ANGLE toward ~90° and raise VISIBILITY_MIN.
const SQUAT_DOWN_ANGLE = 110 // knee angle below this = counted as "down"
const SQUAT_UP_ANGLE = 160 // knee angle above this = counted as "up" (locked out)
const LUNGE_DOWN_ANGLE = 110
const LUNGE_UP_ANGLE = 160

// Wrist speed is measured in normalized units/sec (landmark coords are 0-1).
// These thresholds assume a roughly upper-body-framed shot; they're a
// starting point, not a calibrated constant — see the module comment above.
const STRIKE_EXTEND_SPEED = 1.4
const STRIKE_RETRACT_SPEED = 0.5

const SQUAT_FEEDBACK: Record<Tone, string[]> = {
  good: ["Solid depth on that rep.", "Nice — full range of motion that time."],
  warn: ["Try to squat a touch deeper next rep.", "That one was a bit shallow — aim for thighs closer to parallel."],
  info: ["Keep your chest up and core braced as you descend."],
}
const LUNGE_FEEDBACK: Record<Tone, string[]> = {
  good: ["Good depth on that lunge.", "Nice full range on that rep."],
  warn: ["Lengthen your stride a little for more depth.", "Try to get that front thigh closer to parallel."],
  info: ["Push through the front heel as you rise."],
}
const STRIKE_FEEDBACK: Record<Tone, string[]> = {
  good: ["Quick extension on that one.", "Good snap — nice retraction speed too."],
  warn: ["Reset your guard faster after extending.", "Keep your other hand up while you strike."],
  info: ["Stay light on your feet between strikes."],
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

export class ExerciseAnalyzer {
  private leftLeg: SquatLegState = this.freshLegState()
  private rightLeg: SquatLegState = this.freshLegState()
  private leftWrist: WristState = this.freshWristState()
  private rightWrist: WristState = this.freshWristState()

  private freshLegState(): SquatLegState {
    return {
      phase: "up",
      angleAvg: new RollingAverage(5),
      minAngleThisRep: 180,
    }
  }

  private freshWristState(): WristState {
    return { prevPos: null, prevT: null, speedAvg: new RollingAverage(4), armed: false, peakSpeed: 0 }
  }

  /** Call when the exercise mode changes or a session restarts, so stale phase/averages don't bleed across modes. */
  reset() {
    this.leftLeg = this.freshLegState()
    this.rightLeg = this.freshLegState()
    this.leftWrist = this.freshWristState()
    this.rightWrist = this.freshWristState()
  }

  processFrame(landmarks: Landmark[] | undefined, mode: ExerciseMode, timestampMs: number): FrameResult {
    if (!landmarks || landmarks.length < 33) {
      return { repCompleted: false, formScore: null, cue: null, trackingOk: false }
    }

    switch (mode) {
      case "Squats":
        return this.processSquat(landmarks)
      case "Lunges":
        return this.processLunge(landmarks)
      case "Shadowboxing":
      case "Jab-Cross":
        return this.processStrike(landmarks, timestampMs)
      default: {
        // Exhaustiveness guard: if ExerciseMode ever grows a new variant,
        // this line fails to compile until it's handled above.
        const _exhaustive: never = mode
        return _exhaustive
      }
    }
  }

  // --- Squats: both legs move together, so we track one knee angle (the
  // more visible side). Form is scored on depth only — see note above on
  // why torso lean isn't scored against a fixed reference. ---
  private processSquat(lm: Landmark[]): FrameResult {
    const side = this.pickVisibleSide(lm, L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)
    if (!side) return { repCompleted: false, formScore: null, cue: null, trackingOk: false }

    const { hip, knee, ankle } = side
    const leg = side.isLeft ? this.leftLeg : this.rightLeg
    const rawAngle = angleAt(hip, knee, ankle)
    const angle = leg.angleAvg.push(rawAngle)
    leg.minAngleThisRep = Math.min(leg.minAngleThisRep, angle)

    let repCompleted = false
    let formScore: number | null = null
    let cue: FrameResult["cue"] = null

    if (leg.phase === "up" && angle < SQUAT_DOWN_ANGLE) {
      leg.phase = "down"
    } else if (leg.phase === "down" && angle > SQUAT_UP_ANGLE) {
      leg.phase = "up"
      repCompleted = true

      // Depth score: a smaller minimum knee angle means a deeper squat.
      // Range chosen so the SQUAT_DOWN_ANGLE threshold itself (the minimum
      // depth that counts as a rep at all) scores 60, and a full deep squat
      // (~70° interior angle, ~110° of flexion) scores 100.
      formScore = Math.round(clamp(mapRange(leg.minAngleThisRep, 70, SQUAT_DOWN_ANGLE, 100, 60), 0, 100))

      const tone: Tone = formScore >= 85 ? "good" : formScore >= 65 ? "info" : "warn"
      cue = { text: pick(SQUAT_FEEDBACK[tone]), tone }
      leg.minAngleThisRep = 180
    }

    return { repCompleted, formScore, cue, trackingOk: true }
  }

  // --- Lunges: legs move asymmetrically — track both knees independently,
  // a rep completes whenever either leg finishes a down→up cycle. ---
  private processLunge(lm: Landmark[]): FrameResult {
    const leftOk = visible(lm[L_HIP]) && visible(lm[L_KNEE]) && visible(lm[L_ANKLE])
    const rightOk = visible(lm[R_HIP]) && visible(lm[R_KNEE]) && visible(lm[R_ANKLE])
    if (!leftOk && !rightOk) return { repCompleted: false, formScore: null, cue: null, trackingOk: false }

    let repCompleted = false
    let formScore: number | null = null
    let cue: FrameResult["cue"] = null

    for (const [ok, hipIdx, kneeIdx, ankleIdx, leg] of [
      [leftOk, L_HIP, L_KNEE, L_ANKLE, this.leftLeg],
      [rightOk, R_HIP, R_KNEE, R_ANKLE, this.rightLeg],
    ] as const) {
      if (!ok) continue
      const hip = lm[hipIdx], knee = lm[kneeIdx], ankle = lm[ankleIdx]
      const angle = leg.angleAvg.push(angleAt(hip, knee, ankle))
      leg.minAngleThisRep = Math.min(leg.minAngleThisRep, angle)

      if (leg.phase === "up" && angle < LUNGE_DOWN_ANGLE) {
        leg.phase = "down"
      } else if (leg.phase === "down" && angle > LUNGE_UP_ANGLE) {
        leg.phase = "up"
        repCompleted = true
        const depthScore = clamp(mapRange(leg.minAngleThisRep, 70, LUNGE_DOWN_ANGLE, 100, 60), 0, 100)
        formScore = Math.round(depthScore)
        const tone: Tone = formScore >= 85 ? "good" : formScore >= 65 ? "info" : "warn"
        cue = { text: pick(LUNGE_FEEDBACK[tone]), tone }
        leg.minAngleThisRep = 180
      }
    }

    return { repCompleted, formScore, cue, trackingOk: true }
  }

  // --- Shadowboxing / Jab-Cross: velocity-spike strike detection. Not an
  // established technique like the angle-based squat logic — see module
  // comment. Tracks wrist speed; a "strike" is an extend-then-retract pair. ---
  private processStrike(lm: Landmark[], timestampMs: number): FrameResult {
    let repCompleted = false
    let formScore: number | null = null
    let cue: FrameResult["cue"] = null
    let anyTracked = false

    for (const [wristIdx, wrist] of [
      [L_WRIST, this.leftWrist],
      [R_WRIST, this.rightWrist],
    ] as const) {
      const point = lm[wristIdx]
      if (!visible(point)) {
        wrist.prevPos = null
        wrist.prevT = null
        continue
      }
      anyTracked = true

      if (wrist.prevPos !== null && wrist.prevT !== null) {
        const dt = (timestampMs - wrist.prevT) / 1000
        if (dt > 0) {
          const dist = Math.hypot(point.x - wrist.prevPos.x, point.y - wrist.prevPos.y)
          const speed = wrist.speedAvg.push(dist / dt)

          if (!wrist.armed && speed > STRIKE_EXTEND_SPEED) {
            wrist.armed = true
            wrist.peakSpeed = speed
          } else if (wrist.armed) {
            wrist.peakSpeed = Math.max(wrist.peakSpeed, speed)
            if (speed < STRIKE_RETRACT_SPEED) {
              wrist.armed = false
              repCompleted = true

              // Form proxy: faster peak extension + guard hand (other wrist)
              // staying up near shoulder height both read as "good".
              const speedScore = clamp(mapRange(wrist.peakSpeed, STRIKE_EXTEND_SPEED, 3.5, 60, 100), 0, 100)
              const otherWristIdx = wristIdx === L_WRIST ? R_WRIST : L_WRIST
              const otherShoulderIdx = wristIdx === L_WRIST ? R_SHOULDER : L_SHOULDER
              const guardLm = lm[otherWristIdx]
              const guardShoulderLm = lm[otherShoulderIdx]
              const guardUp = visible(guardLm) && visible(guardShoulderLm) && guardLm.y <= guardShoulderLm.y + 0.08
              const guardScore = guardUp ? 100 : 55
              formScore = Math.round(speedScore * 0.7 + guardScore * 0.3)

              const tone: Tone = formScore >= 85 ? "good" : formScore >= 65 ? "info" : "warn"
              cue = { text: pick(STRIKE_FEEDBACK[tone]), tone }
              wrist.peakSpeed = 0
            }
          }
        }
      }

      wrist.prevPos = { x: point.x, y: point.y }
      wrist.prevT = timestampMs
    }

    return { repCompleted, formScore, cue, trackingOk: anyTracked }
  }

  private pickVisibleSide(
    lm: Landmark[],
    lHip: number, lKnee: number, lAnkle: number,
    rHip: number, rKnee: number, rAnkle: number,
  ) {
    const leftOk = visible(lm[lHip]) && visible(lm[lKnee]) && visible(lm[lAnkle])
    const rightOk = visible(lm[rHip]) && visible(lm[rKnee]) && visible(lm[rAnkle])
    if (leftOk) {
      return { hip: lm[lHip], knee: lm[lKnee], ankle: lm[lAnkle], isLeft: true as const }
    }
    if (rightOk) {
      return { hip: lm[rHip], knee: lm[rKnee], ankle: lm[rAnkle], isLeft: false as const }
    }
    return null
  }
}

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v))
}

function mapRange(v: number, inMin: number, inMax: number, outMin: number, outMax: number) {
  const t = (v - inMin) / (inMax - inMin)
  return outMin + clamp(t, 0, 1) * (outMax - outMin)
}
