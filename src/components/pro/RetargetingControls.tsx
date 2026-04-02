import React from "react";
import { motion } from "framer-motion";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { RotateCw, Eye, Smile, Gauge } from "lucide-react";

interface RetargetingControlsProps {
  headPose: { yaw: number; pitch: number; roll: number };
  onHeadPoseChange: (axis: "yaw" | "pitch" | "roll", value: number) => void;
  gazeDirection: { x: number; y: number };
  onGazeChange: (axis: "x" | "y", value: number) => void;
  expressionOverride: string;
  onExpressionChange: (expr: string) => void;
  emotionIntensity: number;
  onEmotionIntensityChange: (v: number) => void;
  enabled: boolean;
  onEnabledChange: (v: boolean) => void;
}

const EXPRESSIONS = [
  { id: "neutral", label: "Neutral", emoji: "😐" },
  { id: "happy", label: "Happy", emoji: "😊" },
  { id: "surprised", label: "Surprised", emoji: "😮" },
  { id: "angry", label: "Angry", emoji: "😠" },
  { id: "sad", label: "Sad", emoji: "😢" },
];

const RetargetingControls: React.FC<RetargetingControlsProps> = ({
  headPose,
  onHeadPoseChange,
  gazeDirection,
  onGazeChange,
  expressionOverride,
  onExpressionChange,
  emotionIntensity,
  onEmotionIntensityChange,
  enabled,
  onEnabledChange,
}) => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <RotateCw className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Face Retargeting</h3>
        </div>
        <Switch checked={enabled} onCheckedChange={onEnabledChange} />
      </div>

      {enabled && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="space-y-5 pt-2"
        >
          {/* Head Pose */}
          <div className="space-y-3">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
              <RotateCw className="w-3 h-3" /> Head Pose
            </Label>
            {(["yaw", "pitch", "roll"] as const).map((axis) => (
              <div key={axis} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground capitalize">{axis}</span>
                  <span className="text-xs font-mono text-primary">{headPose[axis]}°</span>
                </div>
                <Slider
                  value={[headPose[axis]]}
                  onValueChange={([v]) => onHeadPoseChange(axis, v)}
                  min={-45}
                  max={45}
                  step={1}
                />
              </div>
            ))}
          </div>

          {/* Gaze Direction */}
          <div className="space-y-3">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
              <Eye className="w-3 h-3" /> Eye Gaze
            </Label>
            {(["x", "y"] as const).map((axis) => (
              <div key={axis} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground">{axis === "x" ? "Horizontal" : "Vertical"}</span>
                  <span className="text-xs font-mono text-primary">{gazeDirection[axis]}</span>
                </div>
                <Slider
                  value={[gazeDirection[axis]]}
                  onValueChange={([v]) => onGazeChange(axis, v)}
                  min={-30}
                  max={30}
                  step={1}
                />
              </div>
            ))}
          </div>

          {/* Expression Override */}
          <div className="space-y-3">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
              <Smile className="w-3 h-3" /> Expression
            </Label>
            <div className="grid grid-cols-5 gap-1.5">
              {EXPRESSIONS.map((expr) => (
                <button
                  key={expr.id}
                  onClick={() => onExpressionChange(expr.id)}
                  className={`flex flex-col items-center gap-1 p-2 rounded-lg border transition-all text-center ${
                    expressionOverride === expr.id
                      ? "border-primary bg-primary/10 shadow-[0_0_8px_hsl(var(--primary)/0.2)]"
                      : "border-border bg-secondary hover:border-muted-foreground"
                  }`}
                >
                  <span className="text-lg">{expr.emoji}</span>
                  <span className="text-[9px] text-muted-foreground">{expr.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Emotion Intensity */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-foreground flex items-center gap-1.5">
                <Gauge className="w-3 h-3" /> Intensity
              </Label>
              <span className="text-xs font-mono text-primary">{emotionIntensity}%</span>
            </div>
            <Slider
              value={[emotionIntensity]}
              onValueChange={([v]) => onEmotionIntensityChange(v)}
              min={0}
              max={100}
              step={5}
            />
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default RetargetingControls;
