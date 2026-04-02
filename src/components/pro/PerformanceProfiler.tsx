import React from "react";
import { motion } from "framer-motion";
import { BarChart3, Clock, HardDrive, Cpu } from "lucide-react";
import { Progress } from "@/components/ui/progress";

export interface StageProfile {
  stage: string;
  duration_ms: number;
  vram_peak_mb: number;
  gpu_util_peak: number;
}

interface PerformanceProfilerProps {
  stages: StageProfile[];
  totalTime: number;
}

const PerformanceProfiler: React.FC<PerformanceProfilerProps> = ({ stages, totalTime }) => {
  if (stages.length === 0) return null;

  const maxDuration = Math.max(...stages.map((s) => s.duration_ms));
  const maxVram = Math.max(...stages.map((s) => s.vram_peak_mb));

  const formatMs = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  // Find bottleneck
  const bottleneck = stages.reduce((a, b) => (a.duration_ms > b.duration_ms ? a : b));

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-border bg-card p-4 space-y-4"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Performance Profile</h3>
        </div>
        <span className="text-xs font-mono text-muted-foreground flex items-center gap-1">
          <Clock className="w-3 h-3" /> {formatMs(totalTime)}
        </span>
      </div>

      {/* Stage breakdown */}
      <div className="space-y-2">
        {stages.map((stage) => {
          const isBottleneck = stage === bottleneck;
          const pct = (stage.duration_ms / totalTime) * 100;
          return (
            <div key={stage.stage} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className={`font-medium ${isBottleneck ? "text-warning" : "text-foreground"}`}>
                  {stage.stage}
                  {isBottleneck && (
                    <span className="ml-1.5 text-[9px] font-mono text-warning bg-warning/10 px-1 py-0.5 rounded">
                      BOTTLENECK
                    </span>
                  )}
                </span>
                <span className="font-mono text-muted-foreground">{formatMs(stage.duration_ms)}</span>
              </div>
              <div className="flex items-center gap-2">
                <Progress
                  value={pct}
                  className="h-1.5 flex-1"
                />
                <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground shrink-0">
                  <span className="flex items-center gap-0.5">
                    <HardDrive className="w-2.5 h-2.5" /> {stage.vram_peak_mb}MB
                  </span>
                  <span className="flex items-center gap-0.5">
                    <Cpu className="w-2.5 h-2.5" /> {stage.gpu_util_peak}%
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
};

export default PerformanceProfiler;
