import type { AdvancedSettings, DetectedFace, FaceAssignment } from "@/store/types";

const API_BASE = "http://127.0.0.1:5001/api";
const BACKEND_ORIGIN = "http://127.0.0.1:5001";

export type ProcessingMode = "fast" | "studio" | "cinematic";
export type UpscaleScale = 2 | 4;

export interface HealthStatus {
  status: string;
  gpu: string;
  cuda: boolean;
  models_loaded: boolean;
}

export interface SwapImageResult {
  success: boolean;
  output_url: string;
  processing_time: number;
  job_id?: string;
  quality_score?: {
    overall: number;
    face_confidence: number;
    sharpness: number;
    blur_detected: boolean;
    artifacts: boolean;
  };
}

export interface BatchResultItem {
  filename: string;
  output_url?: string;
  status: "done" | "error";
  message?: string;
}

export interface BatchResult {
  success: boolean;
  results: BatchResultItem[];
  job_id?: string;
}

export interface VideoResult {
  success: boolean;
  output_url: string;
  processing_time: number;
  job_id?: string;
}

export interface SwapPairsResult {
  success: boolean;
  job_id?: string;
}

export interface DetectFacesResult {
  faces: DetectedFace[];
  faces_count?: number;
  auto_select?: boolean;
  face_count?: number;
  requires_selection?: boolean;
  auto_selected_face_id?: number | null;
  suggested_face_id?: number | null;
  image_width?: number;
  image_height?: number;
}

export function outputUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${BACKEND_ORIGIN}${path}`;
}

function extractFilenameFromPath(pathOrUrl: string): string {
  try {
    const parsed = pathOrUrl.startsWith("http")
      ? new URL(pathOrUrl)
      : new URL(pathOrUrl, BACKEND_ORIGIN);
    const tail = parsed.pathname.split("/").pop();
    return tail || "download";
  } catch {
    const tail = pathOrUrl.split("?")[0].split("/").pop();
    return tail || "download";
  }
}

function extractFilenameFromContentDisposition(header: string | null): string | null {
  if (!header) return null;

  const utf8Match = header.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1]);
    } catch {
      return utf8Match[1];
    }
  }

  const plainMatch = header.match(/filename="?([^";]+)"?/i);
  return plainMatch?.[1] || null;
}

function buildAdvancedFormData(fd: FormData, settings: AdvancedSettings) {
  fd.append("mask_feather", String(settings.maskFeather));
  fd.append("mask_expansion", String(settings.maskExpansion));
  fd.append("blend_strength", String(settings.blendStrength));
  fd.append("morph_ratio", String(settings.morphRatio));
  fd.append("expression_match", String(settings.expressionMatch));
  fd.append("use_target_expression", String(settings.expressionMatch));
  fd.append("swap_mode", settings.expressionMatch ? "adaptive" : "raw_copy");
  fd.append("lighting_match", String(settings.lightingMatch));
  fd.append("quality_score", String(settings.qualityScore));
  fd.append("watermark", String(settings.watermark));
}

export const api = {
  async health(): Promise<HealthStatus> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("Backend unreachable");
    return res.json();
  },

  async detectFaces(image: File): Promise<DetectFacesResult> {
    const fd = new FormData();
    fd.append("image", image);
    const res = await fetch(`${API_BASE}/detect/faces`, { method: "POST", body: fd });
    if (!res.ok) throw new Error("Face detection failed");
    return res.json();
  },

  async swapImage(
    source: File,
    target: File,
    mode: ProcessingMode,
    options: { enhance: boolean; upscale: boolean; scale: UpscaleScale },
    advancedSettings?: AdvancedSettings,
    faceAssignments?: FaceAssignment[],
    selectedFaceId?: number
  ): Promise<SwapImageResult> {
    const fd = new FormData();
    fd.append("source", source);
    fd.append("target", target);
    fd.append("mode", mode);
    fd.append("enhance", String(options.enhance));
    fd.append("upscale", String(options.upscale));
    fd.append("scale", String(options.scale));
    if (advancedSettings) buildAdvancedFormData(fd, advancedSettings);
    if (faceAssignments?.length) fd.append("face_assignments", JSON.stringify(faceAssignments));
    if (selectedFaceId !== undefined) fd.append("selected_face_id", String(selectedFaceId));

    const res = await fetch(`${API_BASE}/swap/image`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Swap failed" }));
      throw new Error(err.error || err.message || "Swap failed");
    }
    return res.json();
  },

  async swapBatch(
    source: File,
    targets: File[],
    mode: ProcessingMode,
    options: { enhance: boolean; upscale: boolean; scale: UpscaleScale },
    advancedSettings?: AdvancedSettings
  ): Promise<BatchResult> {
    const fd = new FormData();
    fd.append("source", source);
    targets.forEach((f) => fd.append("targets", f));
    fd.append("mode", mode);
    fd.append("enhance", String(options.enhance));
    fd.append("upscale", String(options.upscale));
    fd.append("scale", String(options.scale));
    if (advancedSettings) buildAdvancedFormData(fd, advancedSettings);

    const res = await fetch(`${API_BASE}/swap/batch`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Batch failed" }));
      throw new Error(err.error || err.message || "Batch failed");
    }
    return res.json();
  },

  async swapVideo(
    source: File,
    targetVideo: File,
    mode: ProcessingMode,
    options: { enhance: boolean; upscale: boolean },
    advancedSettings?: AdvancedSettings
  ): Promise<VideoResult> {
    const fd = new FormData();
    fd.append("source", source);
    fd.append("target_video", targetVideo);
    fd.append("mode", mode);
    fd.append("enhance", String(options.enhance));
    fd.append("upscale", String(options.upscale));
    if (advancedSettings) {
      buildAdvancedFormData(fd, advancedSettings);
      fd.append("keyframe_detection", String(advancedSettings.keyframeDetection));
      fd.append("frame_skip", String(advancedSettings.frameSkip));
    }

    const res = await fetch(`${API_BASE}/swap/video`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Video swap failed" }));
      throw new Error(err.error || err.message || "Video swap failed");
    }
    return res.json();
  },

  async swapPairs(
    pairs: Array<{ source: File; target: File; selectedFaceId?: number }>,
    mode: ProcessingMode,
    options: { enhance: boolean; upscale: boolean; scale: UpscaleScale },
    advancedSettings?: AdvancedSettings
  ): Promise<SwapPairsResult> {
    const fd = new FormData();
    fd.append("pair_count", String(pairs.length));
    fd.append("mode", mode);
    fd.append("enhance", String(options.enhance));
    fd.append("upscale", String(options.upscale));
    fd.append("scale", String(options.scale));
    if (advancedSettings) buildAdvancedFormData(fd, advancedSettings);

    pairs.forEach((pair, idx) => {
      fd.append(`source_${idx}`, pair.source);
      fd.append(`target_${idx}`, pair.target);
      if (pair.selectedFaceId !== undefined) {
        fd.append(`selected_face_id_${idx}`, String(pair.selectedFaceId));
      }
    });

    const res = await fetch(`${API_BASE}/swap/pairs`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Pair batch failed" }));
      throw new Error(err.error || err.message || "Pair batch failed");
    }
    return res.json();
  },

  async cancelJob(jobId: string): Promise<void> {
    await fetch(`${API_BASE}/jobs/${jobId}/cancel`, { method: "POST" });
  },

  async resetState(jobId?: string): Promise<void> {
    await fetch(`${API_BASE}/reset/state`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId || null }),
    });
  },

  async downloadFile(url: string, filename: string): Promise<void> {
    // Extract just the basename from any path/URL
    // If filename is provided, use it; otherwise extract from URL
    let downloadFilename = filename;
    if (!downloadFilename || downloadFilename === "result.png") {
      downloadFilename = extractFilenameFromPath(url);
    }
    // Make sure we only have the basename, not the full path
    const baseName = downloadFilename.split('/').pop() || downloadFilename;
    const encodedName = encodeURIComponent(baseName);
    const downloadUrl = `${API_BASE}/download/${encodedName}`;

    console.log(`[Download] URL: ${url}, Filename: ${baseName}, Request: ${downloadUrl}`);

    const response = await fetch(downloadUrl, { method: "GET" });
    if (!response.ok) {
      const contentType = response.headers.get("content-type") || "";
      let err: any = { error: "Download failed" };
      if (contentType.includes("application/json")) {
        try {
          err = await response.json();
        } catch {}
      }
      throw new Error(err.error || err.message || `Download failed (${response.status})`);
    }

    const contentDisposition = response.headers.get("content-disposition");
    const serverFilename = extractFilenameFromContentDisposition(contentDisposition);
    const resolvedFilename = serverFilename || baseName;

    const blob = await response.blob();
    const blobUrl = window.URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = resolvedFilename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    window.setTimeout(() => {
      window.URL.revokeObjectURL(blobUrl);
    }, 1000);
  }
};
