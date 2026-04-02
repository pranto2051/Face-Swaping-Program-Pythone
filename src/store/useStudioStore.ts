import type { AdvancedSettings, DetectedFace, FaceAssignment, GpuStats, CpuStats, RamStats, ProgressData, Job, QualityScoreData, JobStatus } from "./types";
import { DEFAULT_ADVANCED_SETTINGS } from "./types";
import { create } from "zustand";

export interface RetargetingState {
  enabled: boolean;
  headPose: { yaw: number; pitch: number; roll: number };
  gazeDirection: { x: number; y: number };
  expressionOverride: string;
  emotionIntensity: number;
}

export interface ColorGradingState {
  enabled: boolean;
  selectedLut: string;
  filmGrain: number;
  noiseHarmonize: boolean;
  bgToneMatch: boolean;
}

export interface DiffusionState {
  enabled: boolean;
  strength: number;
  steps: number;
}

interface StudioState {
  // Results
  results: any[];
  singleResult: string | null;
  processingTime: number | null;
  processing: boolean;
  showJobQueue: boolean;
  setResults: (results: any[]) => void;
  setSingleResult: (res: string | null) => void;
  setProcessingTime: (t: number | null) => void;
  setProcessing: (p: boolean) => void;
  setShowJobQueue: (v: boolean) => void;
  batchZipUrl: string | null;
  setBatchZipUrl: (url: string | null) => void;

  // GPU
  gpuStats: GpuStats | null;
  setGpuStats: (stats: GpuStats | null) => void;

  // CPU
  cpuStats: CpuStats | null;
  setCpuStats: (stats: CpuStats | null) => void;

  // RAM
  ramStats: RamStats | null;
  setRamStats: (stats: RamStats | null) => void;

  // Progress
  progress: Record<string, ProgressData>;
  setProgress: (jobId: string, data: ProgressData) => void;
  clearProgress: (jobId: string) => void;

  // Jobs
  jobs: Job[];
  addJob: (job: Job) => void;
  updateJobStatus: (jobId: string, status: JobStatus) => void;
  removeJob: (jobId: string) => void;
  addJobResult: (jobId: string, resultUrl: string | any[]) => void;

  // Face Detection
  detectedFaces: DetectedFace[];
  setDetectedFaces: (faces: DetectedFace[]) => void;
  faceAssignments: FaceAssignment[];
  setFaceAssignments: (assignments: FaceAssignment[]) => void;
  toggleFaceSelection: (targetFaceId: number) => void;
  clearFaceSelection: () => void;

  // Advanced Settings
  advancedSettings: AdvancedSettings;
  updateAdvancedSetting: <K extends keyof AdvancedSettings>(key: K, value: AdvancedSettings[K]) => void;
  resetAdvancedSettings: () => void;

  // Quality Score
  qualityScore: QualityScoreData | null;
  setQualityScore: (score: QualityScoreData | null) => void;

  // Batch max
  batchMaxFiles: number;
  setBatchMaxFiles: (n: number) => void;

  // Socket
  socketConnected: boolean;
  setSocketConnected: (v: boolean) => void;

  // Cancellation
  cancelJob: (jobId: string) => Promise<void>;

  // V3: Retargeting
  retargeting: RetargetingState;
  setRetargeting: (partial: Partial<RetargetingState>) => void;

  // V3: Color Grading
  colorGrading: ColorGradingState;
  setColorGrading: (partial: Partial<ColorGradingState>) => void;

  // V3: Diffusion
  diffusion: DiffusionState;
  setDiffusion: (partial: Partial<DiffusionState>) => void;
}

export const useStudioStore = create<StudioState>((set) => ({
  // Results
  results: [],
  singleResult: null,
  processingTime: null,
  processing: false,
  showJobQueue: true,
  batchZipUrl: null,
  setResults: (results) => set({ results }),
  setSingleResult: (singleResult) => set({ singleResult }),
  setProcessingTime: (processingTime) => set({ processingTime }),
  setProcessing: (processing) => set({ processing }),
  setShowJobQueue: (showJobQueue) => set({ showJobQueue }),
  setBatchZipUrl: (batchZipUrl) => set({ batchZipUrl }),

  // GPU
  gpuStats: null,
  setGpuStats: (stats) => set({ gpuStats: stats }),

  // CPU
  cpuStats: null,
  setCpuStats: (stats) => set({ cpuStats: stats }),

  // RAM
  ramStats: null,
  setRamStats: (stats) => set({ ramStats: stats }),

  // Progress
  progress: {},
  setProgress: (jobId, data) =>
    set((s) => ({ 
      progress: { ...s.progress, [jobId]: data },
      jobs: s.jobs.map((j) => (j.id === jobId ? { ...j, progress: data } : j))
    })),
  clearProgress: (jobId) =>
    set((s) => {
      const { [jobId]: _, ...rest } = s.progress;
      return { progress: rest };
    }),

  // Jobs
  jobs: [],
  addJob: (job) => set((s) => ({ jobs: [job, ...s.jobs] })),
  updateJobStatus: (jobId, status) =>
    set((s) => ({
      jobs: s.jobs.map((j) => (j.id === jobId ? { ...j, status } : j)),
    })),
  removeJob: (jobId) => set((s) => ({ jobs: s.jobs.filter((j) => j.id !== jobId) })),
  addJobResult: (jobId, resultUrl) =>
    set((s) => {
      const job = s.jobs.find((j) => j.id === jobId);
      if (!job) return s;

      if (job.type === "batch") {
        // Handle multiple results (array) for batch jobs
        if (Array.isArray(resultUrl)) {
          return {
            results: [...s.results, ...resultUrl.map((r) => ({ ...r, status: "done" }))],
          };
        } else {
          // Handle single result (string) for backward compatibility
          return {
            results: [...s.results, { filename: job.filename || "result", output_url: resultUrl, status: "done" }],
          };
        }
      } else {
        return { singleResult: resultUrl };
      }
    }),

  // Face Detection
  detectedFaces: [],
  setDetectedFaces: (faces) => set({ detectedFaces: faces }),
  faceAssignments: [],
  setFaceAssignments: (assignments) => set({ faceAssignments: assignments }),
  toggleFaceSelection: (targetFaceId) =>
    set(() => ({
      // Single-face mode: always keep exactly one selected assignment.
      faceAssignments: [{ targetFaceId, sourceFaceId: 0 }],
    })),
  clearFaceSelection: () => set({ detectedFaces: [], faceAssignments: [] }),

  // Advanced Settings
  advancedSettings: DEFAULT_ADVANCED_SETTINGS,
  updateAdvancedSetting: (key, value) =>
    set((s) => ({
      advancedSettings: { ...s.advancedSettings, [key]: value },
    })),
  resetAdvancedSettings: () => set({ advancedSettings: DEFAULT_ADVANCED_SETTINGS }),

  // Quality Score
  qualityScore: null,
  setQualityScore: (score) => set({ qualityScore: score }),

  // Batch max
  batchMaxFiles: 10,
  setBatchMaxFiles: (n) => set({ batchMaxFiles: n }),

  // Socket
  socketConnected: false,
  setSocketConnected: (v) => set({ socketConnected: v }),

  // Cancellation
  cancelJob: async (jobId) => {
    try {
      await fetch(`http://127.0.0.1:5001/api/jobs/${jobId}/cancel`, { method: "POST" });
      set((s) => ({
        jobs: s.jobs.map((j) => (j.id === jobId ? { ...j, status: "cancelled" } : j)),
        progress: {
          ...s.progress,
          [jobId]: { ...s.progress[jobId], percent: 0, stage: "idle" as const },
        },
      }));
    } catch (err) {
      console.error("Failed to cancel job:", err);
    }
  },

  // V3: Retargeting
  retargeting: {
    enabled: false,
    headPose: { yaw: 0, pitch: 0, roll: 0 },
    gazeDirection: { x: 0, y: 0 },
    expressionOverride: "neutral",
    emotionIntensity: 50,
  },
  setRetargeting: (partial) =>
    set((s) => ({ retargeting: { ...s.retargeting, ...partial } })),

  // V3: Color Grading
  colorGrading: {
    enabled: false,
    selectedLut: "none",
    filmGrain: 0,
    noiseHarmonize: true,
    bgToneMatch: true,
  },
  setColorGrading: (partial) =>
    set((s) => ({ colorGrading: { ...s.colorGrading, ...partial } })),

  // V3: Diffusion
  diffusion: {
    enabled: false,
    strength: 30,
    steps: 15,
  },
  setDiffusion: (partial) =>
    set((s) => ({ diffusion: { ...s.diffusion, ...partial } })),
}));
