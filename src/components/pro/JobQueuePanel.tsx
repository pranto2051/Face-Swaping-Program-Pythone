import React from "react";
import { motion } from "framer-motion";
import { ListOrdered, X, Loader2, CheckCircle2, AlertCircle, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { useStudioStore } from "@/store/useStudioStore";
import type { JobStatus } from "@/store/types";

const STATUS_CONFIG: Record<JobStatus, { icon: React.ElementType; color: string; label: string }> = {
  queued: { icon: Clock, color: "text-muted-foreground", label: "Queued" },
  processing: { icon: Loader2, color: "text-primary", label: "Processing" },
  completed: { icon: CheckCircle2, color: "text-success", label: "Completed" },
  failed: { icon: AlertCircle, color: "text-destructive", label: "Failed" },
  cancelled: { icon: X, color: "text-muted-foreground", label: "Cancelled" },
};

const JobQueuePanel: React.FC = () => {
  const { jobs, removeJob, updateJobStatus } = useStudioStore();

  if (jobs.length === 0) return null;

  const handleCancel = async (jobId: string) => {
    try {
      await fetch(`http://127.0.0.1:5001/api/jobs/${jobId}/cancel`, { method: "POST" });
      updateJobStatus(jobId, "cancelled");
    } catch {
      // silent
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-border bg-card overflow-hidden w-72"
    >
      <div className="flex items-center gap-2 p-3 border-b border-border">
        <ListOrdered className="w-4 h-4 text-primary" />
        <h3 className="text-sm font-semibold text-foreground">Job Queue</h3>
        <span className="text-[10px] font-mono text-muted-foreground ml-auto">
          {jobs.filter((j) => j.status === "processing").length} active
        </span>
      </div>

      <ScrollArea className="max-h-48">
        <div className="divide-y divide-border">
          {jobs.map((job) => {
            const config = STATUS_CONFIG[job.status];
            const Icon = config.icon;
            const showProgress = job.status === "processing" && job.progress;
            const percent = job.progress?.percent || 0;
            const stage = job.progress?.stage || "";
            const currentItem = job.progress?.current_item;
            const totalItems = job.progress?.total_items;
            
            return (
              <div key={job.id} className="px-3 py-2.5">
                <div className="flex items-center gap-3">
                  <Icon
                    className={`w-4 h-4 shrink-0 ${config.color} ${job.status === "processing" ? "animate-spin" : ""}`}
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-foreground truncate">
                      {job.filename || job.id.slice(0, 8)}
                    </p>
                    <p className="text-[10px] text-muted-foreground font-mono capitalize">{job.type}</p>
                  </div>
                  <Badge
                    variant="outline"
                    className={`text-[10px] ${config.color} border-current/30`}
                  >
                    {config.label}
                  </Badge>
                  {(job.status === "queued" || job.status === "processing") && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                      onClick={() => handleCancel(job.id)}
                    >
                      <X className="w-3 h-3 text-muted-foreground" />
                    </Button>
                  )}
                  {(job.status === "completed" || job.status === "failed" || job.status === "cancelled") && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                      onClick={() => removeJob(job.id)}
                    >
                      <X className="w-3 h-3 text-muted-foreground" />
                    </Button>
                  )}
                </div>
                
                {/* Progress Bar */}
                {showProgress && (
                  <div className="mt-2 space-y-1">
                    <Progress value={percent} className="h-1.5" />
                    <div className="flex items-center justify-between text-[10px] font-mono">
                      <span className="text-muted-foreground capitalize">{stage}</span>
                      <span className="text-primary font-semibold">
                        {percent}%
                        {currentItem !== undefined && totalItems !== undefined && (
                          <span className="ml-1 text-muted-foreground">
                            ({currentItem}/{totalItems})
                          </span>
                        )}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </motion.div>
  );
};

export default JobQueuePanel;
