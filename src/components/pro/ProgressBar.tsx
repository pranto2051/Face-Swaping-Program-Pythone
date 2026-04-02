import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Progress } from "@/components/ui/progress";
import { useStudioStore } from "@/store/useStudioStore";
import { Clock, Loader2, XCircle, CheckCircle2, AlertCircle, StopCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ProcessingStage, JobStatus } from "@/store/types";

const STAGE_LABELS: Record<ProcessingStage, string> = {
  idle: "Idle",
  detecting: "Detecting Faces",
  aligning: "Aligning",
  swapping: "Swapping Faces",
  refining: "DFL Refinement",
  color_matching: "Color Matching",
  masking: "Mask Refinement",
  enhancing: "GFPGAN Enhancing",
  upscaling: "Real-ESRGAN Upscaling",
  encoding: "Encoding Video",
  done: "Complete",
};

interface ProgressBarProps {
  jobId?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ jobId }) => {
  const { progress, jobs, cancelJob, removeJob } = useStudioStore();

  // Get the target active progress
  const activeProgress = jobId ? progress[jobId] : Object.values(progress).find(p => p.stage !== 'idle');
  const job = jobs.find((j) => j.id === (jobId || activeProgress?.job_id));

  if (!activeProgress || !job) return null;

  const status = job.status;

  const getStatusConfig = (status: JobStatus) => {
    switch (status) {
      case "completed":
        return { color: "text-green-500", bg: "bg-green-500/10", border: "border-green-500/20", icon: CheckCircle2 };
      case "failed":
        return { color: "text-red-500", bg: "bg-red-500/10", border: "border-red-500/20", icon: AlertCircle };
      case "cancelled":
        return { color: "text-orange-500", bg: "bg-orange-500/10", border: "border-orange-500/20", icon: StopCircle };
      default:
        return { color: "text-primary", bg: "bg-primary/5", border: "border-primary/20", icon: Loader2 };
    }
  };

  const config = getStatusConfig(status);
  const Icon = config.icon;

  const formatEta = (seconds: number) => {
    if (seconds < 60) return `${Math.ceil(seconds)}s`;
    return `${Math.floor(seconds / 60)}m ${Math.ceil(seconds % 60)}s`;
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className={`rounded-lg border ${config.border} ${config.bg} p-4 space-y-3 relative overflow-hidden`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Icon className={`w-4 h-4 ${config.color} ${status === 'processing' ? 'animate-spin' : ''}`} />
            <span className="text-sm font-medium text-foreground">
              {status === 'cancelled' ? 'Canceled' : status === 'completed' ? 'Success' : STAGE_LABELS[activeProgress.stage]}
            </span>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
              {activeProgress.current_item != null && activeProgress.total_items != null && (
                <span>
                  {activeProgress.current_item}/{activeProgress.total_items}
                </span>
              )}
              {activeProgress.eta_seconds > 0 && status === 'processing' && (
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {formatEta(activeProgress.eta_seconds)}
                </span>
              )}
            </div>

            {status === 'processing' && (
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-muted-foreground hover:text-orange-500"
                onClick={() => cancelJob(job.id)}
              >
                <XCircle className="w-4 h-4" />
              </Button>
            )}
            
            {(status === 'completed' || status === 'failed' || status === 'cancelled') && (
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-muted-foreground hover:text-destructive"
                onClick={() => removeJob(job.id)}
              >
                <XCircle className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>

        <div className="space-y-1">
          <Progress 
            value={activeProgress.percent} 
            className={`h-2 ${status === 'completed' ? '[&>div]:bg-green-500' : status === 'failed' ? '[&>div]:bg-red-500' : status === 'cancelled' ? '[&>div]:bg-orange-500' : ''}`} 
          />
          <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
            <span>{activeProgress.percent.toFixed(1)}%</span>
            <span className="capitalize">{status}</span>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default ProgressBar;
