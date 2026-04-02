import React from "react";
import { motion } from "framer-motion";
import { Cpu, Thermometer, HardDrive, Activity } from "lucide-react";
import { useStudioStore } from "@/store/useStudioStore";
import { Progress } from "@/components/ui/progress";

const GpuMonitorPanel: React.FC = () => {
  const gpuStats = useStudioStore((s) => s.gpuStats);

  if (!gpuStats) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">GPU Monitor</h3>
        </div>
        <p className="text-xs text-muted-foreground font-mono">Waiting for GPU data...</p>
      </div>
    );
  }

  const tempColor =
    gpuStats.temperature_c > 80
      ? "text-destructive"
      : gpuStats.temperature_c > 65
      ? "text-warning"
      : "text-success";

  const vramColor =
    gpuStats.vram_percent > 90
      ? "text-destructive"
      : gpuStats.vram_percent > 70
      ? "text-warning"
      : "text-primary";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-border bg-card p-4 space-y-3 w-72"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">GPU Monitor</h3>
        </div>
        <span className="text-[10px] font-mono text-muted-foreground">{gpuStats.gpu_name}</span>
      </div>

      {/* GPU Utilization */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <Activity className="w-3 h-3" /> GPU Load
          </span>
          <span className="font-mono text-foreground">{gpuStats.gpu_util_percent}%</span>
        </div>
        <Progress value={gpuStats.gpu_util_percent} className="h-1.5" />
      </div>

      {/* VRAM */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <HardDrive className="w-3 h-3" /> VRAM
          </span>
          <span className={`font-mono ${vramColor}`}>
            {gpuStats.vram_used_mb}MB / {gpuStats.vram_total_mb}MB
          </span>
        </div>
        <Progress value={gpuStats.vram_percent} className="h-1.5" />
      </div>

      {/* Temperature + CUDA */}
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <Thermometer className="w-3 h-3" />
          <span className={`font-mono ${tempColor}`}>{gpuStats.temperature_c}°C</span>
        </span>
        <span className={`font-mono text-xs ${gpuStats.cuda_available ? "text-success" : "text-destructive"}`}>
          {gpuStats.cuda_available ? `CUDA ${gpuStats.cuda_version}` : "No CUDA"}
        </span>
      </div>
    </motion.div>
  );
};

export default GpuMonitorPanel;
