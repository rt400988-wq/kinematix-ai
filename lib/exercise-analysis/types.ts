// Shared types for the exercise-analysis module group. Split out of a single
// 900+ line exercise-analyzer.ts so each concern (types, geometry,
// smoothing, calibration, the analyzer itself) lives in its own file.
//
// Takes MediaPipe's own NormalizedLandmark type directly (rather than a
// parallel custom shape) so there's no structural-compatibility guesswork
// at the call site in app/page.tsx.

import type { NormalizedLandmark } from "@mediapipe/tasks-vision"

export type Landmark = NormalizedLandmark
export type Tone = "good" | "warn" | "info"
export type ExerciseMode = "Shadowboxing" | "Squats" | "Lunges" | "Jab-Cross" | "Push-ups" | "Plank"

/**
 * Real-time, mid-movement correction — fires the instant a form fault is
 * detected, independent of whether/when a rep completes. Distinct from
 * `cue`, which only fires once a rep finishes. A formAlert is meant to
 * interrupt ("you are leaning too far forward — right now"), so the UI
 * should treat it with more urgency / different styling than a post-rep cue.
 */
export type FormAlert = { text: string; tone: "warn"; code: string }

export type FrameResult = {
  repCompleted: boolean
  formScore: number | null // 0-100, this rep's form quality, null if no rep judged yet
  cue: { text: string; tone: Tone } | null
  formAlert: FormAlert | null
  trackingOk: boolean // false if required landmarks aren't visible enough to analyze

  // Plank-only fields. Always present (zeroed) for non-Plank modes so
  // callers don't need mode-specific destructuring.
  holdSeconds: number
  bestHoldSeconds: number
  holdBroken: boolean // true the instant a held position breaks — fire-once edge, not a level
}

// --- MediaPipe BlazePose landmark indices --------------------------------
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
export const NOSE = 0
// Declared for reference completeness alongside the rest of this index
// table, even though nothing in this module currently reads them —
// chin-height proxying uses NOSE instead (see calibration.ts / analyzer.ts).
export const MOUTH_L = 9
export const MOUTH_R = 10
export const L_SHOULDER = 11
export const R_SHOULDER = 12
// Declared for reference completeness; not currently used (push-up/strike
// logic reads shoulder-elbow-wrist angles and wrist velocity directly,
// without needing the elbow index on its own elsewhere).
export const L_ELBOW = 13
export const R_ELBOW = 14
export const L_WRIST = 15
export const R_WRIST = 16
export const L_HIP = 23
export const R_HIP = 24
export const L_KNEE = 25
export const R_KNEE = 26
export const L_ANKLE = 27
export const R_ANKLE = 28

export const VISIBILITY_MIN = 0.5 // "intermediate" tolerance — strict would be ~0.7+
