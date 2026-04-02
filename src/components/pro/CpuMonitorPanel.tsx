import React from "react";
import { motion } from "framer-motion";
import { Cpu, Thermometer, Activity, Layers } from "lucide-react";
import { useStudioStore } from "@/store/useStudioStore";
import { Progress } from "@/components/ui/progress";

const CpuMonitorPanel: React.FC = () => {
  const cpuStats = useStudioStore((s) => s.cpuStats);

  if (!cpuStats) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-accent" />
          <h3 className="text-sm font-semibold text-foreground">CPU Monitor</h3>
        </div>
        <p className="text-xs text-muted-foreground font-mono">Waiting for CPU data...</p>
      </div>
    );
  }

  const tempColor =
    cpuStats.temperature_c > 85
      ? "text-destructive"
      : cpuStats.temperature_c > 70
      ? "text-warning"
      : "text-success";

  const loadColor =
    cpuStats.cpu_percent > 90
      ? "text-destructive"
      : cpuStats.cpu_percent > 70
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
          <Cpu className="w-4 h-4 text-accent" />
          <h3 className="text-sm font-semibold text-foreground">CPU Monitor</h3>
        </div>
        <span className="text-[10px] font-mono text-muted-foreground truncate max-w-[140px]">
          {cpuStats.cpu_name}
        </span>
      </div>

      {/* Overall CPU Load */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <Activity className="w-3 h-3" /> CPU Load
          </span>
          <span className={`font-mono ${loadColor}`}>{cpuStats.cpu_percent}%</span>
        </div>
        <Progress value={cpuStats.cpu_percent} className="h-1.5" />
      </div>

      {/* Per-core mini bars */}
      {cpuStats.per_core_percent.length > 0 && (
        <div className="space-y-1">
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Layers className="w-3 h-3" /> Cores ({cpuStats.core_count}C / {cpuStats.thread_count}T)
          </span>
          <div className="grid grid-cols-4 gap-1">
            {cpuStats.per_core_percent.slice(0, 16).map((pct, i) => (
              <div key={i} className="space-y-0.5">
                <div className="h-1 w-full rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: pct > 90 ? 'hsl(var(--destructive))' : pct > 70 ? 'hsl(var(--warning))' : 'hsl(var(--primary))',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Frequency + Temp */}
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <Activity className="w-3 h-3" />
          <span className="font-mono text-foreground">{(cpuStats.frequency_mhz / 1000).toFixed(2)} GHz</span>
        </span>
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <Thermometer className="w-3 h-3" />
          <span className={`font-mono ${tempColor}`}>{cpuStats.temperature_c}°C</span>
        </span>
      </div>
    </motion.div>
  );
};

export default CpuMonitorPanel;
