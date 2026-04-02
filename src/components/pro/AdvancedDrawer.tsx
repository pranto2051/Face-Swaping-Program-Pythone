import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, SlidersHorizontal, RotateCcw, Layers } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { useStudioStore } from "@/store/useStudioStore";
import RetargetingControls from "./RetargetingControls";
import ColorGradingPanel from "./ColorGradingPanel";
import DiffusionControls from "./DiffusionControls";

interface AdvancedDrawerProps {
  open: boolean;
  onClose: () => void;
  showVideoSettings?: boolean;
  showGpuMonitor?: boolean;
  onShowGpuMonitorChange?: (v: boolean) => void;
  showCpuMonitor?: boolean;
  onShowCpuMonitorChange?: (v: boolean) => void;
  showRamMonitor?: boolean;
  onShowRamMonitorChange?: (v: boolean) => void;
  showJobQueue?: boolean;
  onShowJobQueueChange?: (v: boolean) => void;
}

const AdvancedDrawer: React.FC<AdvancedDrawerProps> = ({ open, onClose, showVideoSettings, showGpuMonitor, onShowGpuMonitorChange, showCpuMonitor, onShowCpuMonitorChange, showRamMonitor, onShowRamMonitorChange, showJobQueue, onShowJobQueueChange }) => {
  const {
    advancedSettings, updateAdvancedSetting, resetAdvancedSettings,
    batchMaxFiles, setBatchMaxFiles,
    retargeting, setRetargeting,
    colorGrading, setColorGrading,
    diffusion, setDiffusion,
  } = useStudioStore();

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-background/60 backdrop-blur-sm z-40"
          />
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-96 bg-card border-l border-border z-50 flex flex-col"
          >
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="w-4 h-4 text-primary" />
                <h2 className="text-sm font-semibold text-foreground">Advanced Settings</h2>
              </div>
              <div className="flex items-center gap-1">
                <Button variant="ghost" size="sm" onClick={resetAdvancedSettings} className="h-7 px-2 text-xs text-muted-foreground">
                  <RotateCcw className="w-3 h-3 mr-1" /> Reset
                </Button>
                <Button variant="ghost" size="icon" onClick={onClose} className="h-7 w-7">
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <ScrollArea className="flex-1">
              <div className="p-4 space-y-6">
                {/* V3: Diffusion Refinement */}
                <DiffusionControls
                  enabled={diffusion.enabled}
                  onEnabledChange={(v) => setDiffusion({ enabled: v })}
                  strength={diffusion.strength}
                  onStrengthChange={(v) => setDiffusion({ strength: v })}
                  steps={diffusion.steps}
                  onStepsChange={(v) => setDiffusion({ steps: v })}
                />

                <Separator className="bg-border" />

                {/* V3: Face Retargeting */}
                <RetargetingControls
                  headPose={retargeting.headPose}
                  onHeadPoseChange={(axis, v) =>
                    setRetargeting({ headPose: { ...retargeting.headPose, [axis]: v } })
                  }
                  gazeDirection={retargeting.gazeDirection}
                  onGazeChange={(axis, v) =>
                    setRetargeting({ gazeDirection: { ...retargeting.gazeDirection, [axis]: v } })
                  }
                  expressionOverride={retargeting.expressionOverride}
                  onExpressionChange={(v) => setRetargeting({ expressionOverride: v })}
                  emotionIntensity={retargeting.emotionIntensity}
                  onEmotionIntensityChange={(v) => setRetargeting({ emotionIntensity: v })}
                  enabled={retargeting.enabled}
                  onEnabledChange={(v) => setRetargeting({ enabled: v })}
                />

                <Separator className="bg-border" />

                {/* V3: Color Grading */}
                <ColorGradingPanel
                  selectedLut={colorGrading.selectedLut}
                  onLutChange={(v) => setColorGrading({ selectedLut: v })}
                  filmGrain={colorGrading.filmGrain}
                  onFilmGrainChange={(v) => setColorGrading({ filmGrain: v })}
                  noiseHarmonize={colorGrading.noiseHarmonize}
                  onNoiseHarmonizeChange={(v) => setColorGrading({ noiseHarmonize: v })}
                  bgToneMatch={colorGrading.bgToneMatch}
                  onBgToneMatchChange={(v) => setColorGrading({ bgToneMatch: v })}
                  enabled={colorGrading.enabled}
                  onEnabledChange={(v) => setColorGrading({ enabled: v })}
                />

                <Separator className="bg-border" />

                {/* Mask Controls */}
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Mask Controls</h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs text-foreground">Feather</Label>
                      <span className="text-xs font-mono text-primary">{advancedSettings.maskFeather}</span>
                    </div>
                    <Slider
                      value={[advancedSettings.maskFeather]}
                      onValueChange={([v]) => updateAdvancedSetting("maskFeather", v)}
                      min={0} max={50} step={1}
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs text-foreground">Expansion</Label>
                      <span className="text-xs font-mono text-primary">{advancedSettings.maskExpansion}</span>
                    </div>
                    <Slider
                      value={[advancedSettings.maskExpansion]}
                      onValueChange={([v]) => updateAdvancedSetting("maskExpansion", v)}
                      min={-20} max={20} step={1}
                    />
                  </div>
                </div>

                <Separator className="bg-border" />

                {/* Blend Controls */}
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Blend Controls</h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs text-foreground">Blend Strength</Label>
                      <span className="text-xs font-mono text-primary">{advancedSettings.blendStrength}%</span>
                    </div>
                    <Slider
                      value={[advancedSettings.blendStrength]}
                      onValueChange={([v]) => updateAdvancedSetting("blendStrength", v)}
                      min={0} max={100} step={1}
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs text-foreground">Morph Ratio</Label>
                      <span className="text-xs font-mono text-primary">{advancedSettings.morphRatio}%</span>
                    </div>
                    <Slider
                      value={[advancedSettings.morphRatio]}
                      onValueChange={([v]) => updateAdvancedSetting("morphRatio", v)}
                      min={0} max={100} step={1}
                    />
                  </div>
                </div>

                <Separator className="bg-border" />

                {/* AI Toggles */}
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">AI Features</h3>
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <Label
                        className="text-xs cursor-pointer text-foreground"
                        title="When OFF, the source face is copied as-is. No smile, eye, or mouth changes are applied."
                      >
                        Face Expression
                      </Label>
                      <Switch
                        checked={advancedSettings.expressionMatch}
                        onCheckedChange={(v) => {
                          updateAdvancedSetting("expressionMatch", v);
                          if (!v) {
                            updateAdvancedSetting("lightingMatch", false);
                          }
                        }}
                      />
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                      {advancedSettings.expressionMatch
                        ? "ON: Adapt to target expression"
                        : "OFF: Preserve source exactly (Raw Copy Mode)"}
                    </p>
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs cursor-pointer text-foreground">Lighting Match</Label>
                    <Switch
                      checked={advancedSettings.lightingMatch}
                      disabled={!advancedSettings.expressionMatch}
                      onCheckedChange={(v) => updateAdvancedSetting("lightingMatch", v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs cursor-pointer text-foreground">Quality Score</Label>
                    <Switch checked={advancedSettings.qualityScore} onCheckedChange={(v) => updateAdvancedSetting("qualityScore", v)} />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs cursor-pointer text-foreground">Watermark</Label>
                    <Switch checked={advancedSettings.watermark} onCheckedChange={(v) => updateAdvancedSetting("watermark", v)} />
                  </div>
                  {advancedSettings.watermark && (
                    <div className="pl-1 space-y-1.5">
                      <Label className="text-xs text-muted-foreground">Watermark Text</Label>
                      <Input
                        value={advancedSettings.watermarkText}
                        onChange={(e) => updateAdvancedSetting("watermarkText", e.target.value)}
                        placeholder="Enter watermark text..."
                        className="h-8 text-xs bg-background"
                      />
                    </div>
                  )}
                </div>

                <Separator className="bg-border" />

                {/* Batch Settings */}
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Batch Settings</h3>
                  <div className="flex items-center justify-between gap-4">
                    <Label className="text-xs text-foreground whitespace-nowrap">Max target images</Label>
                    <Input
                      type="number" min={2} max={100} value={batchMaxFiles}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10);
                        if (!isNaN(v) && v >= 2 && v <= 100) setBatchMaxFiles(v);
                      }}
                      className="w-20 h-7 text-xs bg-background"
                    />
                  </div>
                </div>

                <Separator className="bg-border" />

                {/* Panels Visibility */}
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Panels</h3>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs cursor-pointer text-foreground">GPU Monitor</Label>
                    <Switch
                      checked={showGpuMonitor ?? true}
                      onCheckedChange={(v) => onShowGpuMonitorChange?.(v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs cursor-pointer text-foreground">CPU Monitor</Label>
                    <Switch
                      checked={showCpuMonitor ?? false}
                      onCheckedChange={(v) => onShowCpuMonitorChange?.(v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs cursor-pointer text-foreground">RAM Monitor</Label>
                    <Switch
                      checked={showRamMonitor ?? false}
                      onCheckedChange={(v) => onShowRamMonitorChange?.(v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs cursor-pointer text-foreground">Job Queue</Label>
                    <Switch
                      checked={showJobQueue ?? true}
                      onCheckedChange={(v) => onShowJobQueueChange?.(v)}
                    />
                  </div>
                </div>

                <Separator className="bg-border" />

                {/* Monitoring Layout */}
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Monitor Alignment</h3>
                  <div className="flex p-1 bg-secondary rounded-lg">
                    <Button
                      variant={advancedSettings.monitorLayout === "list" ? "default" : "ghost"}
                      size="sm"
                      onClick={() => updateAdvancedSetting("monitorLayout", "list")}
                      className="flex-1 h-8 text-xs gap-2"
                    >
                      <Layers className="w-3.5 h-3.5" /> Column
                    </Button>
                    <Button
                      variant={advancedSettings.monitorLayout === "grid" ? "default" : "ghost"}
                      size="sm"
                      onClick={() => updateAdvancedSetting("monitorLayout", "grid")}
                      className="flex-1 h-8 text-xs gap-2"
                    >
                      <SlidersHorizontal className="w-3.5 h-3.5 rotate-90" /> Row
                    </Button>
                  </div>
                </div>

                {/* Video Settings */}
                {showVideoSettings && (
                  <>
                    <Separator className="bg-border" />
                    <div className="space-y-4">
                      <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Video Optimization</h3>
                      <div className="flex items-center justify-between">
                        <Label className="text-xs cursor-pointer text-foreground">Keyframe Detection</Label>
                        <Switch checked={advancedSettings.keyframeDetection} onCheckedChange={(v) => updateAdvancedSetting("keyframeDetection", v)} />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label className="text-xs text-foreground">Frame Skip</Label>
                          <span className="text-xs font-mono text-primary">{advancedSettings.frameSkip}</span>
                        </div>
                        <Slider
                          value={[advancedSettings.frameSkip]}
                          onValueChange={([v]) => updateAdvancedSetting("frameSkip", v)}
                          min={1} max={10} step={1}
                        />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label className="text-xs text-foreground">Max Video Capacity</Label>
                          <span className="text-xs font-mono text-primary">{advancedSettings.videoCapacityLimit} MB</span>
                        </div>
                        <Slider
                          value={[advancedSettings.videoCapacityLimit]}
                          onValueChange={([v]) => updateAdvancedSetting("videoCapacityLimit", v)}
                          min={50} max={2048} step={50}
                        />
                      </div>
                    </div>
                  </>
                )}
              </div>
            </ScrollArea>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default AdvancedDrawer;
