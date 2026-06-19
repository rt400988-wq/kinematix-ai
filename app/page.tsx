"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { DashboardHeader } from "@/components/dashboard-header"
import { CameraFeed } from "@/components/camera-feed"
import { MetricsSidebar } from "@/components/metrics-sidebar"
import { AiFeedback, type Feedback } from "@/components/ai-feedback"
import { ExerciseAnalyzer, type ExerciseMode } from "@/lib/exercise-analyzer"
import type { NormalizedLandmark } from "@mediapipe/tasks-vision"

const MODES = ["Shadowboxing", "Squats", "Lunges", "Jab-Cross"] as const

// How many recent rep scores feed the displayed accuracy number. A short
// window keeps the metric responsive to how the last few reps actually
// looked, rather than a lifetime average that barely moves after rep 30.
const ACCURACY_WINDOW = 5

export default function Page() {
  const [active, setActive] = useState(false)
  const [mode, setMode] = useState<ExerciseMode>(MODES[0])
  const [reps, setReps] = useState(0)
  const [accuracy, setAccuracy] = useState(0)
  const [feedback, setFeedback] = useState<Feedback[]>([])
  const [tracking, setTracking] = useState(false) // true once a pose has actually been seen
  const fbId = useRef(0)

  // Lives for the whole session — owns the per-mode state machines (squat
  // phase, wrist velocity, etc). A ref because it's mutated every frame and
  // must NOT be recreated on re-render.
  const analyzerRef = useRef<ExerciseAnalyzer | null>(null)
  if (!analyzerRef.current) analyzerRef.current = new ExerciseAnalyzer()

  // Recent rep form-scores, used to compute the displayed accuracy as a
  // short rolling average (see ACCURACY_WINDOW above) rather than an
  // ever-more-sluggish lifetime average.
  const recentScoresRef = useRef<number[]>([])

  // Reset all per-mode tracking state whenever the exercise mode changes,
  // so e.g. a half-completed squat rep doesn't bleed into a freshly
  // selected Lunges session.
  useEffect(() => {
    analyzerRef.current?.reset()
    recentScoresRef.current = []
    setTracking(false)
  }, [mode])

  // Fired on every detection frame (~30x/sec) via CameraFeed -> usePoseLandmarker.
  // Deliberately does NOT setState on every call — only when the analyzer
  // reports something actually happened (a completed rep/strike), so the
  // dashboard doesn't re-render 30 times a second.
  const handlePoseLandmarks = useCallback(
    (landmarks: NormalizedLandmark[] | undefined, timestampMs: number) => {
      const analyzer = analyzerRef.current
      if (!analyzer) return

      const result = analyzer.processFrame(landmarks, mode, timestampMs)

      if (result.trackingOk) {
        setTracking((prev) => (prev ? prev : true))
      }

      if (result.repCompleted) {
        setReps((r) => r + 1)

        if (result.formScore !== null) {
          const scores = recentScoresRef.current
          scores.push(result.formScore)
          if (scores.length > ACCURACY_WINDOW) scores.shift()
          const avg = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length)
          setAccuracy(avg)
        }

        if (result.cue) {
          fbId.current += 1
          setFeedback((prev) => [{ id: fbId.current, text: result.cue!.text, tone: result.cue!.tone }, ...prev].slice(0, 6))
        }
      }
    },
    [mode],
  )

  // Reset reps & feedback when switching mode mid-session.
  function handleModeChange(next: string) {
    const nextMode = next as ExerciseMode
    setMode(nextMode)
    if (active) {
      setReps(0)
      setAccuracy(0)
      fbId.current += 1
      setFeedback([
        {
          id: fbId.current,
          text: `Switched to ${next}. Calibrating pose model…`,
          tone: "info",
        },
      ])
    }
  }

  function handleToggle(next: boolean) {
    setActive(next)
    if (!next) {
      setFeedback([])
      setReps(0)
      setAccuracy(0)
      setTracking(false)
      analyzerRef.current?.reset()
      recentScoresRef.current = []
    }
  }

  // Punching modes count a fundamentally different, less precise signal
  // (wrist-velocity spikes) than the angle-based squat/lunge rep counter —
  // see lib/exercise-analyzer.ts. Labeling it "Strikes" instead of "Reps"
  // reflects that difference instead of presenting both as equally exact.
  const repLabel = mode === "Shadowboxing" || mode === "Jab-Cross" ? "strikes" : "reps"

  return (
    <main className="min-h-dvh bg-background">
      <div className="mx-auto flex min-h-dvh max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <DashboardHeader active={active} />

        <div className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-[1fr_22rem]">
          {/* Center stage: webcam */}
          <div className="min-h-[60vh] lg:min-h-0">
            <CameraFeed active={active} mode={mode} reps={reps} onToggle={handleToggle} onLandmarks={handlePoseLandmarks} />
          </div>

          {/* Right sidebar */}
          <div className="flex min-h-0 flex-col gap-6">
            <MetricsSidebar
              active={active}
              reps={reps}
              repLabel={repLabel}
              accuracy={accuracy}
              tracking={tracking}
              mode={mode}
              modes={MODES}
              onModeChange={handleModeChange}
            />
            <AiFeedback active={active} items={feedback} />
          </div>
        </div>
      </div>
    </main>
  )
}
