import React from "react";
import { motion } from "framer-motion";
import { Palette, Film, Sparkles } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface ColorGradingPanelProps {
  selectedLut: string;
  onLutChange: (lut: string) => void;
  filmGrain: number;
  onFilmGrainChange: (v: number) => void;
  noiseHarmonize: boolean;
  onNoiseHarmonizeChange: (v: boolean) => void;
  bgToneMatch: boolean;
  onBgToneMatchChange: (v: boolean) => void;
  enabled: boolean;
  onEnabledChange: (v: boolean) => void;
}

const LUT_PRESETS = [
  { id: "none", label: "None", gradient: "from-muted to-muted" },
  { id: "cinematic_warm", label: "Warm Cinema", gradient: "from-amber-900/60 to-orange-800/40" },
  { id: "cinematic_cool", label: "Cool Teal", gradient: "from-cyan-900/60 to-blue-900/40" },
  { id: "noir", label: "Film Noir", gradient: "from-neutral-900 to-neutral-700" },
  { id: "vintage", label: "Vintage", gradient: "from-yellow-900/50 to-amber-800/30" },
  { id: "neon", label: "Neon Night", gradient: "from-purple-900/60 to-pink-900/40" },
  { id: "bleach_bypass", label: "Bleach Bypass", gradient: "from-slate-800/60 to-zinc-700/40" },
  { id: "golden_hour", label: "Golden Hour", gradient: "from-yellow-800/50 to-orange-700/30" },
];

const ColorGradingPanel: React.FC<ColorGradingPanelProps> = ({
  selectedLut,
  onLutChange,
  filmGrain,
  onFilmGrainChange,
  noiseHarmonize,
  onNoiseHarmonizeChange,
  bgToneMatch,
  onBgToneMatchChange,
  enabled,
  onEnabledChange,
}) => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Palette className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Color Grading</h3>
        </div>
        <Switch checked={enabled} onCheckedChange={onEnabledChange} />
      </div>

      {enabled && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="space-y-5 pt-2"
        >
          {/* LUT Presets */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider">
              Cinematic LUT Preset
            </Label>
            <div className="grid grid-cols-4 gap-2">
              {LUT_PRESETS.map((lut) => (
                <button
                  key={lut.id}
                  onClick={() => onLutChange(lut.id)}
                  className={`relative rounded-lg overflow-hidden border-2 transition-all ${
                    selectedLut === lut.id
                      ? "border-primary shadow-[0_0_8px_hsl(var(--primary)/0.3)]"
                      : "border-border hover:border-muted-foreground"
                  }`}
                >
                  <div className={`h-10 bg-gradient-to-br ${lut.gradient}`} />
                  <span className="block text-[9px] text-center py-1 text-muted-foreground bg-card">
                    {lut.label}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Film Grain */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-foreground flex items-center gap-1.5">
                <Film className="w-3 h-3" /> Film Grain
              </Label>
              <span className="text-xs font-mono text-primary">{filmGrain}%</span>
            </div>
            <Slider
              value={[filmGrain]}
              onValueChange={([v]) => onFilmGrainChange(v)}
              min={0}
              max={100}
              step={5}
            />
          </div>

          {/* Toggles */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-xs cursor-pointer text-foreground flex items-center gap-1.5">
                <Sparkles className="w-3 h-3" /> Noise Harmonization
              </Label>
              <Switch checked={noiseHarmonize} onCheckedChange={onNoiseHarmonizeChange} />
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-xs cursor-pointer text-foreground flex items-center gap-1.5">
                <Palette className="w-3 h-3" /> Background Tone Match
              </Label>
              <Switch checked={bgToneMatch} onCheckedChange={onBgToneMatchChange} />
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ColorGradingPanel;
