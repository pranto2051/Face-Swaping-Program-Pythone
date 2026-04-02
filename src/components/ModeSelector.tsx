import React from "react";
import { Zap, Sparkles, Clapperboard } from "lucide-react";
import type { ProcessingMode } from "@/services/api";

interface ModeSelectorProps {
  mode: ProcessingMode;
  onChange: (mode: ProcessingMode) => void;
}

const MODES: { id: ProcessingMode; label: string; desc: string; icon: React.ElementType }[] = [
  { id: "fast", label: "Fast Mode", desc: "InsightFace only • ~2s", icon: Zap },
  { id: "studio", label: "Studio Mode", desc: "+ DeepFaceLab • ~30s", icon: Sparkles },
  { id: "cinematic", label: "Cinematic Mode", desc: "+ Diffusion + Color • ~60s", icon: Clapperboard },
];

const ModeSelector: React.FC<ModeSelectorProps> = ({ mode, onChange }) => {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground">Processing Mode</label>
      <div className="grid grid-cols-3 gap-3">
        {MODES.map((m) => {
          const Icon = m.icon;
          const active = mode === m.id;
          return (
            <button
              key={m.id}
              onClick={() => onChange(m.id)}
              className={`relative p-4 rounded-lg border-2 transition-all duration-300 text-left ${
                active
                  ? "border-primary bg-primary/5 glow-primary-sm"
                  : "border-border hover:border-muted-foreground bg-card"
              }`}
            >
              <Icon className={`w-5 h-5 mb-2 ${active ? "text-primary" : "text-muted-foreground"}`} />
              <p className="text-sm font-semibold text-foreground">{m.label}</p>
              <p className="text-xs text-muted-foreground mt-1">{m.desc}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ModeSelector;
