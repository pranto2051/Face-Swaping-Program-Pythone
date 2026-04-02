import React from "react";
import { motion } from "framer-motion";
import { MemoryStick, HardDrive } from "lucide-react";
import { useStudioStore } from "@/store/useStudioStore";
import { Progress } from "@/components/ui/progress";

const RamMonitorPanel: React.FC = () => {
  const ramStats = useStudioStore((s) => s.ramStats);

  if (!ramStats) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <MemoryStick className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">RAM Monitor</h3>
        </div>
        <p className="text-xs text-muted-foreground font-mono">Waiting for RAM data...</p>
      </div>
    );
  }

  const ramColor =
    ramStats.percent > 90
      ? "text-destructive"
      : ramStats.percent > 75
      ? "text-warning"
      : "text-success";

  const swapColor =
    ramStats.swap_percent > 80
      ? "text-destructive"
      : ramStats.swap_percent > 50
      ? "text-warning"
      : "text-muted-foreground";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-border bg-card p-4 space-y-3 w-72"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MemoryStick className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">RAM Monitor</h3>
        </div>
        <span className="text-[10px] font-mono text-muted-foreground">
          {ramStats.total_gb.toFixed(1)} GB Total
        </span>
      </div>

      {/* RAM Usage */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <MemoryStick className="w-3 h-3" /> Memory
          </span>
          <span className={`font-mono ${ramColor}`}>
            {ramStats.used_gb.toFixed(1)} / {ramStats.total_gb.toFixed(1)} GB
          </span>
        </div>
        <Progress value={ramStats.percent} className="h-1.5" />
      </div>

      {/* Visual memory block grid */}
      <div className="grid grid-cols-10 gap-0.5">
        {Array.from({ length: 20 }).map((_, i) => {
          const filledBlocks = Math.round((ramStats.percent / 100) * 20);
          return (
            <div
              key={i}
              className="h-2 rounded-sm transition-colors"
              style={{
                backgroundColor: i < filledBlocks
                  ? ramStats.percent > 90
                    ? 'hsl(var(--destructive))'
                    : ramStats.percent > 75
                    ? 'hsl(var(--warning))'
                    : 'hsl(var(--primary))'
                  : 'hsl(var(--secondary))',
              }}
            />
          );
        })}
      </div>

      {/* Swap + Available */}
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <HardDrive className="w-3 h-3" />
          <span className="font-mono text-foreground">
            Free: {ramStats.available_gb.toFixed(1)} GB
          </span>
        </span>
        {ramStats.swap_total_gb > 0 && (
          <span className={`font-mono text-xs ${swapColor}`}>
            Swap: {ramStats.swap_used_gb.toFixed(1)}/{ramStats.swap_total_gb.toFixed(1)} GB
          </span>
        )}
      </div>
    </motion.div>
  );
};

export default RamMonitorPanel;
