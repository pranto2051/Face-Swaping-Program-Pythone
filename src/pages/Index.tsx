import React, { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Image, Film, Layers, Play, Download, Clock, SlidersHorizontal, Plus, Trash2, RotateCcw } from "lucide-react";
import { toast } from "sonner";
import DropZone from "@/components/DropZone";
import ModeSelector from "@/components/ModeSelector";
import EnhancementOptions from "@/components/EnhancementOptions";
import ResultCard from "@/components/ResultCard";
import ImageLightbox from "@/components/ImageLightbox";
import StatusBar from "@/components/StatusBar";
import GpuMonitorPanel from "@/components/pro/GpuMonitorPanel";
import CpuMonitorPanel from "@/components/pro/CpuMonitorPanel";
import RamMonitorPanel from "@/components/pro/RamMonitorPanel";
import ProgressBar from "@/components/pro/ProgressBar";
import FaceSelectorOverlay from "@/components/pro/FaceSelectorOverlay";
import AdvancedDrawer from "@/components/pro/AdvancedDrawer";
import QualityScoreBadge from "@/components/pro/QualityScoreBadge";
import JobQueuePanel from "@/components/pro/JobQueuePanel";
import { useSocket } from "@/hooks/useSocket";
import { useGpuMonitor } from "@/hooks/useGpuMonitor";
import { useSystemMonitor } from "@/hooks/useSystemMonitor";
import { useStudioStore } from "@/store/useStudioStore";
import { api, type ProcessingMode, type UpscaleScale, type BatchResultItem } from "@/services/api";
import type { DetectedFace } from "@/store/types";
import heroBg from "@/assets/hero-bg.jpg";

type PairStatus = "waiting" | "processing" | "completed" | "failed";

interface ImagePair {
  id: number;
  sourceFile: File | null;
  targetFile: File | null;
  sourcePreview: string | null;
  targetPreview: string | null;
  faces: DetectedFace[];
  selectedFaceId: number | null;
  selectionRequired: boolean;
  autoSelected: boolean;
  detecting: boolean;
  error: string | null;
  status: PairStatus;
  progress: number;
  message: string;
  resultUrl: string | null;
  resultFilename: string | null;
  imageWidth: number | null;
  imageHeight: number | null;
}

const createPair = (id: number): ImagePair => ({
  id,
  sourceFile: null,
  targetFile: null,
  sourcePreview: null,
  targetPreview: null,
  faces: [],
  selectedFaceId: null,
  selectionRequired: false,
  autoSelected: false,
  detecting: false,
  error: null,
  status: "waiting",
  progress: 0,
  message: "Waiting",
  resultUrl: null,
  resultFilename: null,
  imageWidth: null,
  imageHeight: null,
});

const Index = () => {
  const [tab, setTab] = useState("image");
  const [mode, setMode] = useState<ProcessingMode>("fast");
  const [enhance, setEnhance] = useState(true);
  const [upscale, setUpscale] = useState(false);
  const [scale, setScale] = useState<UpscaleScale>(2);
  const [showGpuMonitor, setShowGpuMonitor] = useState(false);
  const [showCpuMonitor, setShowCpuMonitor] = useState(false);
  const [showRamMonitor, setShowRamMonitor] = useState(false);
  const [pairs, setPairs] = useState<ImagePair[]>([createPair(1)]);
  const [nextPairId, setNextPairId] = useState(2);
  const [pairJobId, setPairJobId] = useState<string | null>(null);
  const [pairIdOrder, setPairIdOrder] = useState<number[]>([]);
  const [sourceFiles, setSourceFiles] = useState<File[]>([]);
  const [targetFiles, setTargetFiles] = useState<File[]>([]);
  const [videoFiles, setVideoFiles] = useState<File[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxIndex, setLightboxIndex] = useState(0);

  // Pro hooks
  useSocket();
  useGpuMonitor();
  useSystemMonitor();

  // Store
  const {
    batchMaxFiles,
    advancedSettings,
    qualityScore,
    setQualityScore,
    progress,
    jobs,
    batchZipUrl,
    setBatchZipUrl,
    retargeting,
    colorGrading,
    diffusion,
    results,
    setResults,
    singleResult,
    setSingleResult,
    processingTime,
    setProcessingTime,
    processing,
    setProcessing,
    showJobQueue,
    setShowJobQueue,
    addJob,
  } = useStudioStore();
  const targetFaceDetectionVersion = useRef(0);

  const handleTabChange = (newTab: string) => {
    pairs.forEach((p) => {
      if (p.sourcePreview) URL.revokeObjectURL(p.sourcePreview);
      if (p.targetPreview) URL.revokeObjectURL(p.targetPreview);
    });
    setTab(newTab);
    setSourceFiles([]);
    setTargetFiles([]);
    setVideoFiles([]);
    setPairs([createPair(1)]);
    setNextPairId(2);
    setPairJobId(null);
    setPairIdOrder([]);
    setResults([]);
    setSingleResult(null);
    setProcessingTime(null);
    setBatchZipUrl(null);
    setQualityScore(null);
  };

  const addPair = useCallback(() => {
    setPairs((prev) => [...prev, createPair(nextPairId)]);
    setNextPairId((v) => v + 1);
  }, [nextPairId]);

  const removePair = useCallback((pairId: number) => {
    setPairs((prev) => {
      const target = prev.find((p) => p.id === pairId);
      if (target?.sourcePreview) URL.revokeObjectURL(target.sourcePreview);
      if (target?.targetPreview) URL.revokeObjectURL(target.targetPreview);

      const next = prev.filter((p) => p.id !== pairId);
      return next.length > 0 ? next : [createPair(1)];
    });
  }, []);

  const resetPairs = useCallback(() => {
    if (pairJobId) {
      api.resetState(pairJobId).catch(() => {
        // Keep reset UX resilient even if cleanup endpoint fails.
      });
    }
    pairs.forEach((p) => {
      if (p.sourcePreview) URL.revokeObjectURL(p.sourcePreview);
      if (p.targetPreview) URL.revokeObjectURL(p.targetPreview);
    });
    setPairs([createPair(1)]);
    setNextPairId(2);
    setPairJobId(null);
    setPairIdOrder([]);
    setBatchZipUrl(null);
    setResults([]);
    setSingleResult(null);
    setProcessing(false);
    setProcessingTime(null);
    setQualityScore(null);
  }, [pairJobId, pairs, setBatchZipUrl, setResults, setSingleResult, setProcessing, setProcessingTime, setQualityScore]);

  const updatePair = useCallback((pairId: number, updater: (pair: ImagePair) => ImagePair) => {
    setPairs((prev) => prev.map((p) => (p.id === pairId ? updater(p) : p)));
  }, []);

  const detectFacesForPair = useCallback(async (pairId: number, targetFile: File) => {
    const requestVersion = ++targetFaceDetectionVersion.current;

    updatePair(pairId, (pair) => ({
      ...pair,
      detecting: true,
      error: null,
      faces: [],
      selectedFaceId: null,
      selectionRequired: false,
      autoSelected: false,
    }));

    try {
      const data = await api.detectFaces(targetFile);
      if (requestVersion !== targetFaceDetectionVersion.current) return;

      const autoSelect = data.auto_select ??
        (data.faces_count !== undefined ? data.faces_count === 1 : data.faces.length === 1);

      updatePair(pairId, (pair) => {
        if (data.faces.length === 0) {
          return {
            ...pair,
            detecting: false,
            faces: [],
            selectedFaceId: null,
            selectionRequired: false,
            autoSelected: false,
            error: "No face detected in target image.",
          };
        }

        if (autoSelect && data.faces.length === 1) {
          return {
            ...pair,
            detecting: false,
            faces: data.faces,
            selectedFaceId: data.faces[0].id,
            selectionRequired: false,
            autoSelected: true,
            imageWidth: data.image_width ?? null,
            imageHeight: data.image_height ?? null,
            error: null,
          };
        }

        return {
          ...pair,
          detecting: false,
          faces: data.faces,
          selectedFaceId: null,
          selectionRequired: data.faces.length > 1,
          autoSelected: false,
          imageWidth: data.image_width ?? null,
          imageHeight: data.image_height ?? null,
          error: data.faces.length > 1 ? "Multiple faces detected. Select one." : null,
        };
      });
    } catch (error: any) {
      updatePair(pairId, (pair) => ({
        ...pair,
        detecting: false,
        error: error?.message || "Face detection failed",
      }));
    }
  }, [updatePair]);

  const setPairSource = useCallback((pairId: number, files: File[]) => {
    const file = files[0] || null;
    setPairs((prev) => prev.map((pair) => {
      if (pair.id !== pairId) return pair;
      if (pair.sourcePreview) URL.revokeObjectURL(pair.sourcePreview);
      return {
        ...pair,
        sourceFile: file,
        sourcePreview: file ? URL.createObjectURL(file) : null,
      };
    }));
  }, []);

  const setPairTarget = useCallback((pairId: number, files: File[]) => {
    const file = files[0] || null;
    setPairs((prev) => prev.map((pair) => {
      if (pair.id !== pairId) return pair;
      if (pair.targetPreview) URL.revokeObjectURL(pair.targetPreview);
      return {
        ...pair,
        targetFile: file,
        targetPreview: file ? URL.createObjectURL(file) : null,
        faces: [],
        selectedFaceId: null,
        selectionRequired: false,
        autoSelected: false,
        error: null,
        resultUrl: null,
        resultFilename: null,
        status: "waiting",
        progress: 0,
        message: "Waiting",
      };
    }));

    if (file) {
      detectFacesForPair(pairId, file);
    }
  }, [detectFacesForPair]);

  const selectFaceForPair = useCallback((pairId: number, faceId: number) => {
    updatePair(pairId, (pair) => ({
      ...pair,
      selectedFaceId: faceId,
      error: null,
    }));
  }, [updatePair]);

  useEffect(() => {
    if (!pairJobId) return;
    const p = progress[pairJobId];
    if (!p?.pairs) return;

    setPairs((prev) => prev.map((pair) => {
      const runtime = p.pairs?.find((x) => {
        const localId = pairIdOrder[x.pair_id - 1] ?? x.pair_id;
        return localId === pair.id;
      });
      if (!runtime) return pair;
      return {
        ...pair,
        status: runtime.status,
        progress: runtime.progress,
        message: runtime.message || pair.message,
        resultUrl: runtime.result_url || pair.resultUrl,
        resultFilename: runtime.filename || pair.resultFilename,
      };
    }));
  }, [progress, pairJobId, pairIdOrder]);

  const handleSwap = useCallback(async () => {
    setProcessing(true);
    setResults([]);
    setSingleResult(null);
    setProcessingTime(null);
    setQualityScore(null);
    setBatchZipUrl(null);

    try {
      let jobId: string | undefined;

      if (tab === "video") {
        if (sourceFiles.length === 0) {
          setProcessing(false);
          return;
        }
        if (videoFiles.length === 0) return;
        const data = await api.swapVideo(sourceFiles[0], videoFiles[0], mode, { enhance, upscale }, advancedSettings);
        jobId = data.job_id;
        if (jobId) {
          addJob({ id: jobId, type: "video", status: "queued", progress: null, createdAt: Date.now() });
          toast.info("Video job queued...");
        }
      } else if (tab === "batch") {
        if (sourceFiles.length === 0) {
          setProcessing(false);
          return;
        }
        if (targetFiles.length === 0) return;
        const data = await api.swapBatch(sourceFiles[0], targetFiles, mode, { enhance, upscale, scale }, advancedSettings);
        jobId = data.job_id;
        if (jobId) {
          addJob({ id: jobId, type: "batch", status: "queued", progress: null, createdAt: Date.now() });
          toast.info(`${targetFiles.length} images queued for background processing...`);
        }
      } else {
        if (pairs.length === 0) {
          toast.warning("Add at least one pair.");
          setProcessing(false);
          return;
        }

        const invalidPair = pairs.find((pair) => {
          if (!pair.sourceFile || !pair.targetFile) return true;
          if (pair.detecting) return true;
          if (pair.error && pair.faces.length === 0) return true;
          if (pair.selectionRequired && pair.selectedFaceId === null) return true;
          return false;
        });

        if (invalidPair) {
          toast.warning(`Pair ${invalidPair.id} is incomplete. Please fix it before processing.`);
          setProcessing(false);
          return;
        }

        setPairs((prev) => prev.map((pair) => ({
          ...pair,
          status: "waiting",
          progress: 0,
          message: "Waiting",
          resultUrl: null,
          resultFilename: null,
        })));

        const payload = pairs.map((pair) => ({
          source: pair.sourceFile!,
          target: pair.targetFile!,
          selectedFaceId: pair.selectedFaceId ?? undefined,
        }));
        setPairIdOrder(pairs.map((pair) => pair.id));

        const data = await api.swapPairs(payload, mode, { enhance, upscale, scale }, advancedSettings);
        jobId = data.job_id;
        if (jobId) {
          setPairJobId(jobId);
          addJob({ id: jobId, type: "batch", status: "queued", progress: null, createdAt: Date.now() });
          toast.info(`Batch started: ${pairs.length} pair(s)`);
        }
      }
    } catch (err: any) {
      toast.error(err.message || "Processing failed");
      setProcessing(false);
    }
  }, [sourceFiles, targetFiles, videoFiles, tab, mode, enhance, upscale, scale, advancedSettings, pairs, setQualityScore, addJob, setResults, setSingleResult, setProcessingTime, setBatchZipUrl]);

  const imageSelectionReady = tab !== "image" || pairs.every((pair) => {
    if (!pair.sourceFile || !pair.targetFile) return false;
    if (pair.detecting) return false;
    if (pair.faces.length === 0) return false;
    if (pair.selectionRequired && pair.selectedFaceId === null) return false;
    if (pair.error && pair.faces.length === 0) return false;
    return true;
  });

  const canProcess =
    (tab === "image"
      ? pairs.length > 0
      : sourceFiles.length > 0 && (tab === "video" ? videoFiles.length > 0 : targetFiles.length > 0)) &&
    imageSelectionReady &&
    !processing;
  
  // Check for active jobs in current tab
  const activeJobForTab = jobs.find(job => 
    job.type === tab && 
    (job.status === "processing" || job.status === "queued")
  );
  
  const isProcessingCurrentTab = !!activeJobForTab || processing;
  const currentProgress = activeJobForTab?.progress;

  const hasResults = results.length > 0 || singleResult;

  return (
    <div className="min-h-screen bg-background">
      {/* Hero */}
      <div className="relative h-[280px] overflow-hidden">
        <img src={heroBg} alt="" className="absolute inset-0 w-full h-full object-cover opacity-40" />
        <div className="absolute inset-0 bg-gradient-to-b from-background/30 via-background/60 to-background" />
        <div className="relative z-10 h-full flex flex-col items-center justify-center px-4">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-5xl font-bold tracking-tight text-gradient-primary"
          >
            DeepFake Studio
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mt-3 text-muted-foreground text-sm md:text-base max-w-md text-center"
          >
            Professional AI face swap • InsightFace + DeepFaceLab hybrid engine
          </motion.p>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="mt-4">
            <StatusBar />
          </motion.div>
        </div>
      </div>

      {/* Main */}
      <div className="max-w-7xl mx-auto px-4 -mt-6 pb-16 relative">
        {/* Floating GPU/Job panels - top right */}
        <AnimatePresence>
          {(showGpuMonitor || showCpuMonitor || showRamMonitor || showJobQueue) && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className={`fixed top-4 right-4 z-30 max-h-[90vh] overflow-y-auto scrollbar-thin transition-all duration-300 ${
                advancedSettings.monitorLayout === "grid" 
                  ? "w-auto max-w-[90vw] flex flex-row-reverse flex-wrap gap-3 items-start justify-end" 
                  : "w-72 flex flex-col gap-3"
              }`}
            >
              {[
                showJobQueue && <JobQueuePanel key="queue" />,
                showRamMonitor && <RamMonitorPanel key="ram" />,
                showCpuMonitor && <CpuMonitorPanel key="cpu" />,
                showGpuMonitor && <GpuMonitorPanel key="gpu" />,
              ].filter(Boolean)}
            </motion.div>
          )}
        </AnimatePresence>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="rounded-xl border border-border bg-card/80 backdrop-blur-sm p-6 md:p-8"
        >
            <Tabs value={tab} onValueChange={handleTabChange} className="space-y-6">
              <div className="flex items-center justify-between">
                <TabsList className="bg-secondary border border-border">
                  <TabsTrigger value="image" className="gap-2 data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
                    <Image className="w-4 h-4" /> Image
                  </TabsTrigger>
                  <TabsTrigger value="batch" className="gap-2 data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
                    <Layers className="w-4 h-4" /> Batch
                  </TabsTrigger>
                  <TabsTrigger value="video" className="gap-2 data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
                    <Film className="w-4 h-4" /> Video
                  </TabsTrigger>
                </TabsList>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setDrawerOpen(true)}
                  className="gap-2 text-xs border-border"
                >
                  <SlidersHorizontal className="w-3.5 h-3.5" /> Advanced
                </Button>
              </div>

              <TabsContent value="image" className="space-y-6">
                <div className="flex flex-wrap items-center gap-2">
                  <Button variant="outline" size="sm" className="gap-2" onClick={addPair} disabled={processing}>
                    <Plus className="w-4 h-4" /> Add Pair
                  </Button>
                  <Button variant="outline" size="sm" className="gap-2" onClick={resetPairs} disabled={processing}>
                    <RotateCcw className="w-4 h-4" /> Reset
                  </Button>
                  <span className="text-xs text-muted-foreground font-mono">{pairs.length} pair(s) ready</span>
                </div>

                <div className="space-y-4">
                  {pairs.map((pair) => (
                    <motion.div
                      key={pair.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="rounded-xl border border-border bg-secondary/40 shadow-sm p-4 space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold">Pair {pair.id}</span>
                          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                            pair.status === "processing" ? "bg-blue-500/15 text-blue-600" :
                            pair.status === "completed" ? "bg-green-500/15 text-green-600" :
                            pair.status === "failed" ? "bg-red-500/15 text-red-600" :
                            "bg-orange-500/15 text-orange-600"
                          }`}>
                            {pair.status.toUpperCase()}
                          </span>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          disabled={processing || pairs.length === 1}
                          onClick={() => removePair(pair.id)}
                          className="h-7 w-7"
                        >
                          <Trash2 className="w-4 h-4 text-muted-foreground" />
                        </Button>
                      </div>

                      <div className="grid md:grid-cols-2 gap-4">
                        <DropZone
                          label="Source Image"
                          files={pair.sourceFile ? [pair.sourceFile] : []}
                          onFiles={(files) => setPairSource(pair.id, files)}
                          compact
                        />
                        <DropZone
                          label="Target Image"
                          files={pair.targetFile ? [pair.targetFile] : []}
                          onFiles={(files) => setPairTarget(pair.id, files)}
                          compact
                        />
                      </div>

                      {pair.detecting && (
                        <p className="text-xs text-blue-600">Detecting faces...</p>
                      )}

                      {!pair.detecting && pair.autoSelected && (
                        <p className="text-xs text-green-600">1 Face Detected - Auto Selected</p>
                      )}

                      {!pair.detecting && pair.error && pair.faces.length === 0 && (
                        <p className="text-xs text-red-600">{pair.error}</p>
                      )}

                      {!pair.detecting && pair.selectionRequired && pair.targetPreview && pair.faces.length > 1 && (
                        <div className="space-y-2">
                          <p className="text-xs text-amber-600">Multiple faces detected. Select one.</p>
                          <div className="rounded-lg border border-border p-2 bg-card">
                            <div className="relative inline-block max-w-full">
                              <img src={pair.targetPreview} alt={`Pair ${pair.id} target`} className="block max-h-72 w-auto max-w-full object-contain rounded" />
                              {pair.imageWidth && pair.imageHeight && pair.faces.map((face) => {
                                const [x1, y1, x2, y2] = face.bbox;
                                const left = (x1 / pair.imageWidth) * 100;
                                const top = (y1 / pair.imageHeight) * 100;
                                const width = ((x2 - x1) / pair.imageWidth) * 100;
                                const height = ((y2 - y1) / pair.imageHeight) * 100;
                                const selected = pair.selectedFaceId === face.id;
                                return (
                                  <button
                                    key={face.id}
                                    style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                                    className={`absolute border-2 rounded ${selected ? "border-blue-500 bg-blue-500/15" : "border-muted-foreground/60 hover:border-blue-400"}`}
                                    onClick={() => selectFaceForPair(pair.id, face.id)}
                                    type="button"
                                    disabled={processing}
                                  >
                                    <span className={`absolute -top-2 -left-2 text-[10px] px-1.5 py-0.5 rounded ${selected ? "bg-blue-500 text-white" : "bg-background text-foreground"}`}>
                                      Face {face.id + 1}
                                    </span>
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="space-y-1">
                        <div className="w-full h-1.5 rounded bg-muted overflow-hidden">
                          <div className={`h-full transition-all duration-300 ${
                            pair.status === "completed" ? "bg-green-500" :
                            pair.status === "failed" ? "bg-red-500" :
                            pair.status === "processing" ? "bg-blue-500" :
                            "bg-orange-400"
                          }`} style={{ width: `${pair.progress}%` }} />
                        </div>
                        <p className="text-[11px] text-muted-foreground font-mono">
                          {pair.message}
                          {pairJobId && pair.status === "processing" ? ` - Processing Pair ${pair.id} of ${pairs.length}` : ""}
                        </p>
                      </div>

                      {pair.resultUrl && (
                        <div className="flex items-center justify-between gap-2 rounded-md border border-border p-2 bg-card">
                          <span className="text-xs text-muted-foreground truncate">{pair.resultFilename || `pair_${pair.id}.jpg`}</span>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => api.downloadFile(pair.resultUrl!, pair.resultFilename || `pair_${pair.id}.jpg`)}
                            className="gap-1"
                          >
                            <Download className="w-3 h-3" /> Download
                          </Button>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="batch" className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <DropZone label="Source Face" files={sourceFiles} onFiles={setSourceFiles} />
                  <DropZone label={`Target Images (up to ${batchMaxFiles})`} files={targetFiles} onFiles={setTargetFiles} multiple maxFiles={batchMaxFiles} />
                </div>
              </TabsContent>

              <TabsContent value="video" className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <DropZone label="Source Face" files={sourceFiles} onFiles={setSourceFiles} />
                  <DropZone label="Target Video" accept="video/*" files={videoFiles} onFiles={setVideoFiles} />
                </div>
              </TabsContent>

              <div className="grid md:grid-cols-2 gap-6">
                <ModeSelector mode={mode} onChange={setMode} />
                <EnhancementOptions
                  enhance={enhance}
                  upscale={upscale}
                  scale={scale}
                  onEnhanceChange={setEnhance}
                  onUpscaleChange={setUpscale}
                  onScaleChange={setScale}
                />
              </div>

              {/* Progress Bar */}
              <AnimatePresence>
                {isProcessingCurrentTab && currentProgress && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-2 px-4 py-3 rounded-lg border border-primary/30 bg-primary/5"
                  >
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-foreground capitalize">
                        {currentProgress.stage === "idle" ? "Initializing" : currentProgress.stage.replace(/_/g, " ")}
                      </span>
                      <span className="font-mono text-primary font-semibold">
                        {currentProgress.percent}%
                      </span>
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2 overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-primary to-primary/80 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${currentProgress.percent}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    {currentProgress.current_item !== undefined && currentProgress.total_items !== undefined && (
                      <p className="text-xs text-muted-foreground font-mono text-center">
                        Frame {currentProgress.current_item} of {currentProgress.total_items}
                      </p>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="flex items-center justify-between pt-2">
                <p className="text-xs text-muted-foreground font-mono">
                  {tab === "image" && `${pairs.length} pair(s)`}
                  {tab !== "image" && sourceFiles.length > 0 && `Source: ${sourceFiles[0].name}`}
                  {tab !== "image" && targetFiles.length > 0 && ` • ${targetFiles.length} target(s)`}
                  {videoFiles.length > 0 && ` • Video: ${videoFiles[0].name}`}
                </p>
                <div className="flex items-center gap-2">
                  {tab === "image" && batchZipUrl && (
                    <Button
                      variant="outline"
                      onClick={() => api.downloadFile(batchZipUrl, `batch_${pairJobId || "results"}.zip`)}
                      className="gap-2"
                    >
                      <Download className="w-4 h-4" /> Download All ZIP
                    </Button>
                  )}
                  <Button
                    onClick={handleSwap}
                    disabled={!canProcess || isProcessingCurrentTab}
                    className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90 glow-primary-sm px-6 min-w-[140px]"
                  >
                    {isProcessingCurrentTab ? (
                      <>
                        <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                        <span>
                          {currentProgress?.status === "queued" ? "Queued..." : 
                           currentProgress ? `${currentProgress.percent}%` : "Processing..."}
                        </span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        {tab === "image" ? "Start Processing" : "Start Swap"}
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </Tabs>

            {/* Results */}
            <AnimatePresence>
              {hasResults && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="mt-8 pt-6 border-t border-border space-y-4"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <h2 className="text-lg font-semibold text-foreground">Results</h2>
                      {processingTime !== null && (
                        <span className="flex items-center gap-1 text-xs font-mono text-muted-foreground">
                          <Clock className="w-3 h-3" /> {processingTime.toFixed(1)}s
                        </span>
                      )}
                    </div>
                    {(singleResult || results.some((r) => r.status === "done")) && (
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="gap-2 text-sm border-border" 
                        disabled={downloading}
                        onClick={async () => {
                          if (!singleResult) return;
                          setDownloading(true);
                          try {
                            const filename = singleResult.split('/').pop() || "result.png";
                            await api.downloadFile(singleResult, filename);
                            toast.success("Download Successful ✅", {
                              description: "Your file has been saved to the Downloads folder.",
                              duration: 4000,
                            });
                          } catch (err: any) {
                            toast.error("Download Failed ❌", {
                              description: err?.message || "File not found or inaccessible.",
                            });
                          } finally {
                            setDownloading(false);
                          }
                        }}
                      >
                        {downloading ? (
                          <div className="w-3.5 h-3.5 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                        ) : (
                          <Download className="w-3.5 h-3.5" />
                        )}
                        Download
                      </Button>
                    )}
                  </div>

                  {singleResult && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="rounded-lg border border-border overflow-hidden bg-secondary max-w-lg mx-auto cursor-pointer"
                      onClick={() => {
                        setLightboxIndex(0);
                        setLightboxOpen(true);
                      }}
                    >
                      {tab === "video" ? (
                        <video src={singleResult} controls className="w-full" />
                      ) : (
                        <img src={singleResult} alt="Result" className="w-full hover:opacity-90 transition-opacity" />
                      )}
                    </motion.div>
                  )}

                  {/* Quality Score */}
                  {qualityScore && <QualityScoreBadge score={qualityScore} />}

                  {results.length > 0 && (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                      {results.map((item, i) => (
                        <ResultCard
                          key={item.filename + i}
                          item={item}
                          index={i}
                          onClick={() => {
                            const successItems = results.filter((r) => r.output_url);
                            const lightboxIdx = successItems.findIndex((r) => r.filename === item.filename && r.output_url === item.output_url);
                            if (lightboxIdx >= 0) {
                              setLightboxIndex(lightboxIdx);
                              setLightboxOpen(true);
                            }
                          }}
                        />
                      ))}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

        <div className="mt-8 text-center">
          <p className="text-xs text-muted-foreground font-mono">
            Flask backend at 127.0.0.1:5001 • SocketIO connected • Auto-refresh GPU every 5s
          </p>
        </div>
      </div>

      {/* Advanced Settings Drawer */}
      <AdvancedDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        showVideoSettings={tab === "video"}
        showGpuMonitor={showGpuMonitor}
        onShowGpuMonitorChange={setShowGpuMonitor}
        showCpuMonitor={showCpuMonitor}
        onShowCpuMonitorChange={setShowCpuMonitor}
        showRamMonitor={showRamMonitor}
        onShowRamMonitorChange={setShowRamMonitor}
        showJobQueue={showJobQueue}
        onShowJobQueueChange={setShowJobQueue}
      />

      {/* Fullscreen Image Lightbox */}
      <ImageLightbox
        items={(() => {
          const lightboxItems: { url: string; filename: string }[] = [];
          if (singleResult && tab !== "video") {
            const singleFilename = singleResult.split('/').pop() || "result.png";
            lightboxItems.push({ url: singleResult, filename: singleFilename });
          }
          results
            .filter((r) => r.output_url)
            .forEach((r) => lightboxItems.push({ url: r.output_url!, filename: r.filename }));
          return lightboxItems;
        })()}
        currentIndex={lightboxIndex}
        open={lightboxOpen}
        onClose={() => setLightboxOpen(false)}
        onIndexChange={setLightboxIndex}
      />
    </div>
  );
};

export default Index;
