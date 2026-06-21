// Noise-rejection primitives shared across the analyzer and calibration
// controller. None of these know anything about exercises or landmarks —
// they're generic signal-smoothing building blocks.

// Small moving average so a single noisy frame can't flip a state machine
// and produce a phantom rep, or a phantom form-fault flash. Window of 5 at
// ~30fps is ~165ms — enough to smooth jitter without meaningfully lagging
// real movement.
export class RollingAverage {
  private buf: number[] = []
  constructor(private size = 5) {}
  push(v: number): number {
    this.buf.push(v)
    if (this.buf.length > this.size) this.buf.shift()
    return this.buf.reduce((a, b) => a + b, 0) / this.buf.length
  }
  get sampleCount() {
    return this.buf.length
  }
  reset() {
    this.buf = []
  }
}

// Fires a boolean condition only after it has been continuously true for
// `persistFrames` consecutive checks, and clears the instant the condition
// goes false. Edge-triggered: returns true exactly once, on the frame the
// count crosses the threshold — suited to one-shot events like "flash a
// warning." This is what keeps the multi-joint form alerts (knee valgus,
// back sag, dropped guard) from flickering on a single noisy frame — given
// MediaPipe's published ~19° error band on derived angles like knee valgus,
// a single-frame trigger would fire constantly on noise alone.
export class PersistentFlag {
  private count = 0
  constructor(private persistFrames = 6) {}
  update(condition: boolean): boolean {
    if (condition) {
      this.count++
    } else {
      this.count = 0
    }
    return this.count === this.persistFrames // fires exactly once, on the frame it crosses the threshold
  }
  reset() {
    this.count = 0
  }
}

// Same noise-rejection idea as PersistentFlag, but level-triggered rather
// than edge-triggered: once the condition has held for `persistFrames`
// frames, this stays true for as long as the condition keeps holding (not
// just on the single frame it crosses the threshold). Needed wherever the
// caller treats the result as an ongoing state rather than a one-shot event
// — e.g. "is the plank hold currently broken," which must stay broken for
// as long as misalignment continues, not flip back to "fine" the very next
// frame just because the internal counter ticked past an exact value.
export class PersistentLevel {
  private count = 0
  constructor(private persistFrames = 6) {}
  update(condition: boolean): boolean {
    if (condition) {
      this.count = Math.min(this.count + 1, this.persistFrames)
    } else {
      this.count = 0
    }
    return this.count >= this.persistFrames
  }
  reset() {
    this.count = 0
  }
}

// Time-based persistence flag — same idea as PersistentFlag but driven by
// elapsed video timestamp rather than frame count. Used by
// CalibrationController where steps are measured in "hold this for ~1
// second" rather than "N consecutive frames" (frame rate can vary with
// device load, so a frame-count threshold would be inconsistent across
// devices for these slower, coarser checks).
export class PersistentFlagSeconds {
  private trueSinceMs: number | null = null
  constructor(private requiredSeconds: number) {}
  update(condition: boolean, timestampMs: number): boolean {
    if (!condition) {
      this.trueSinceMs = null
      return false
    }
    if (this.trueSinceMs === null) this.trueSinceMs = timestampMs
    return (timestampMs - this.trueSinceMs) / 1000 >= this.requiredSeconds
  }
  reset() {
    this.trueSinceMs = null
  }
}
