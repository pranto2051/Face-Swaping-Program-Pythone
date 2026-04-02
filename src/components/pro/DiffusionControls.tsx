import React from "react";
import { Sparkles } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface DiffusionControlsProps {
  enabled: boolean;
  onEnabledChange: (v: boolean) => void;
  strength: number;
  onStrengthChange: (v: number) => void;
  steps: number;
  onStepsChange: (v: number) => void;
}

const DiffusionControls: React.FC<DiffusionControlsProps> = ({
  enabled,
  onEnabledChange,
  strength,
  onStrengthChange,
  steps,
  onStepsChange,
}) => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Diffusion Refinement</h3>
        </div>
        <Switch checked={enabled} onCheckedChange={onEnabledChange} />
      </div>

      {enabled && (
        <div className="space-y-4 pt-1">
          <div className="rounded-md bg-primary/5 border border-primary/20 p-2.5">
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              AI-powered skin texture refinement using CodeFormer. Higher strength improves realism but may reduce identity fidelity.
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-foreground">Refinement Strength</Label>
              <span className="text-xs font-mono text-primary">{strength}%</span>
            </div>
            <Slider
              value={[strength]}
              onValueChange={([v]) => onStrengthChange(v)}
              min={0}
              max={100}
              step={5}
            />
            <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
              <span>Identity</span>
              <span>Quality</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-foreground">Denoising Steps</Label>
              <span className="text-xs font-mono text-primary">{steps}</span>
            </div>
            <Slider
              value={[steps]}
              onValueChange={([v]) => onStepsChange(v)}
              min={5}
              max={30}
              step={5}
            />
            <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
              <span>Fast</span>
              <span>Detailed</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DiffusionControls;
