// Pure geometry/math helpers, no exercise-specific logic. Kept dependency-
// free (only needs the Landmark type) so these are easy to unit-test in
// isolation from the state machines that use them.

import type { Landmark } from "./types"
import { VISIBILITY_MIN } from "./types"

export function angleAt(a: Landmark, vertex: Landmark, b: Landmark): number {
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

// Angle of the line a→b from horizontal, in degrees, 0-180. Used for
// straight-line checks (shoulder-hip-ankle for plank/push-ups) where what
// matters is "is this line straight" rather than a joint bend angle.
export function lineAngleFromHorizontal(a: Landmark, b: Landmark): number {
  const dx = b.x - a.x
  const dy = b.y - a.y
  return (Math.atan2(Math.abs(dy), Math.abs(dx)) * 180) / Math.PI
}

export function visible(lm: Landmark | undefined): lm is Landmark {
  return !!lm && (lm.visibility ?? 1) >= VISIBILITY_MIN
}

export function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v))
}

export function mapRange(v: number, inMin: number, inMax: number, outMin: number, outMax: number) {
  const t = (v - inMin) / (inMax - inMin)
  return outMin + clamp(t, 0, 1) * (outMax - outMin)
}

export function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

export function midpoint(a: Landmark, b: Landmark): Landmark {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
  }
}

// Degrees the shoulder-hip-ankle path deviates from a straight line — 0 means
// perfectly straight. Used for plank holds and push-up back-sag/pike checks.
export function backAlignmentDeviation(shoulder: Landmark, hip: Landmark, ankle: Landmark): number {
  const angleAtHip = angleAt(shoulder, hip, ankle)
  return Math.abs(180 - angleAtHip)
}

// Signed version: positive means the hip sits BELOW where a straight
// shoulder-ankle line would put it (sagging), negative means above (piking).
// Magnitude is in the same normalized units as landmark y-coordinates, not
// degrees — only the sign matters here, the angle itself still comes from
// backAlignmentDeviation.
export function hipSagDirection(shoulder: Landmark, hip: Landmark, ankle: Landmark): number {
  const t = ankle.x !== shoulder.x ? clamp((hip.x - shoulder.x) / (ankle.x - shoulder.x), 0, 1) : 0.5
  const expectedHipY = shoulder.y + t * (ankle.y - shoulder.y)
  return hip.y - expectedHipY // image y increases downward, so positive = hip is lower = sagging
}
