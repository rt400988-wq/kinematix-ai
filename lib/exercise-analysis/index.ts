// Barrel export — lets consumers (app/page.tsx, components/camera-feed.tsx)
// import everything from "@/lib/exercise-analysis" without needing to know
// it's internally split across types.ts / geometry.ts / smoothing.ts /
// calibration.ts / analyzer.ts. Geometry and smoothing internals are
// intentionally NOT re-exported here — they're implementation details of
// the analyzer/calibration modules, not part of the public surface other
// files should depend on.

export type { Landmark, Tone, ExerciseMode, FormAlert, FrameResult } from "./types"
export { CalibrationController, type CalibrationStatus, type CalibrationStepId } from "./calibration"
export { ExerciseAnalyzer } from "./analyzer"
