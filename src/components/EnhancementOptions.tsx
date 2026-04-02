import React from "react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Wand2, Maximize } from "lucide-react";
import type { UpscaleScale } from "@/services/api";

interface EnhancementOptionsProps {
  enhance: boolean;
  upscale: boolean;
  scale: UpscaleScale;
  onEnhanceChange: (v: boolean) => void;
  onUpscaleChange: (v: boolean) => void;
  onScaleChange: (v: UpscaleScale) => void;
}

const EnhancementOptions: React.FC<EnhancementOptionsProps> = ({
  enhance,
  upscale,
  scale,
  onEnhanceChange,
  onUpscaleChange,
  onScaleChange,
}) => {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground">Enhancement</label>
      <div className="space-y-3 p-4 rounded-lg bg-card border border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wand2 className="w-4 h-4 text-primary" />
            <Label htmlFor="enhance" className="text-sm cursor-pointer">GFPGAN Face Restore</Label>
          </div>
          <Switch id="enhance" checked={enhance} onCheckedChange={onEnhanceChange} />
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Maximize className="w-4 h-4 text-primary" />
            <Label htmlFor="upscale" className="text-sm cursor-pointer">Real-ESRGAN Upscale</Label>
          </div>
          <Switch id="upscale" checked={upscale} onCheckedChange={onUpscaleChange} />
        </div>
        {upscale && (
          <div className="flex items-center gap-2 pl-6">
            <span className="text-xs text-muted-foreground">Scale:</span>
            <div className="flex gap-1">
              {([2, 4] as UpscaleScale[]).map((s) => (
                <button
                  key={s}
                  onClick={() => onScaleChange(s)}
                  className={`px-3 py-1 text-xs rounded-md font-mono transition-all ${
                    scale === s
                      ? "bg-primary/15 text-primary border border-primary/30"
                      : "bg-secondary text-muted-foreground border border-border hover:border-muted-foreground"
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancementOptions;
