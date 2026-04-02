export type ProcessingMode = "fast" | "studio" | "cinematic";
export type UpscaleScale = 2 | 4;
export type JobStatus = "queued" | "processing" | "completed" | "failed" | "cancelled";
export type ProcessingStage = "idle" | "detecting" | "aligning" | "swapping" | "refining" | "color_matching" | "masking" | "enhancing" | "upscaling" | "encoding" | "done";

export interface DetectedFace {
  id: number;
  bbox: [number, number, number, number]; // x1, y1, x2, y2 in original image pixels
  confidence: number;
}

export interface FaceAssignment {
  targetFaceId: number;
  sourceFaceId: number;
}

export interface AdvancedSettings {
  maskFeather: number;      // 0-50
  maskExpansion: number;     // -20 to +20
  blendStrength: number;    // 0-100
  morphRatio: number;       // 0-100
  expressionMatch: boolean; // true=adaptive expression, false=raw copy strict mode
  lightingMatch: boolean;
  qualityScore: boolean;
  watermark: boolean;
  watermarkText: string;
  // Video-specific
  keyframeDetection: boolean;
  frameSkip: number;        // 1-10
  videoCapacityLimit: number; // 50-2048 MB
  monitorLayout: "grid" | "list";
}

export interface GpuStats {
  gpu_name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  vram_percent: number;
  gpu_util_percent: number;
  temperature_c: number;
  cuda_available: boolean;
  cuda_version: string;
}

export interface CpuStats {
  cpu_name: string;
  cpu_percent: number;
  core_count: number;
  thread_count: number;
  frequency_mhz: number;
  temperature_c: number;
  per_core_percent: number[];
}

export interface RamStats {
  total_gb: number;
  used_gb: number;
  available_gb: number;
  percent: number;
  swap_total_gb: number;
  swap_used_gb: number;
  swap_percent: number;
}

export interface ProgressData {
  job_id: string;
  stage: ProcessingStage;
  percent: number;
  eta_seconds: number;
  current_item?: number;
  total_items?: number;
  status?: JobStatus;
  pairs?: Array<{
    pair_id: number;
    status: "waiting" | "processing" | "completed" | "failed";
    progress: number;
    message?: string;
    result_url?: string | null;
    filename?: string | null;
  }>;
}

export interface QualityScoreData {
  overall: number;        // 0-100
  face_confidence: number;
  sharpness: number;
  blur_detected: boolean;
  artifacts: boolean;
}

export interface Job {
  id: string;
  type: "image" | "batch" | "video";
  status: JobStatus;
  progress: ProgressData | null;
  createdAt: number;
  filename?: string;
}

export const DEFAULT_ADVANCED_SETTINGS: AdvancedSettings = {
  maskFeather: 10,
  maskExpansion: 0,
  blendStrength: 80,
  morphRatio: 50,
  expressionMatch: true,
  lightingMatch: false,
  qualityScore: false,
  watermark: false,
  watermarkText: "DeepFake Studio",
  keyframeDetection: true,
  frameSkip: 3,
  videoCapacityLimit: 2048,
  monitorLayout: "grid",
};
