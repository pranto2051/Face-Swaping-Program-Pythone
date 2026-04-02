import React from "react";
import { motion } from "framer-motion";
import { Shield, Eye, Scan, AlertTriangle } from "lucide-react";
import type { QualityScoreData } from "@/store/types";

interface QualityScoreBadgeProps {
  score: QualityScoreData;
}

const QualityScoreBadge: React.FC<QualityScoreBadgeProps> = ({ score }) => {
  const getColor = (value: number) => {
    if (value >= 80) return { text: "text-success", bg: "bg-success/10 border-success/30", label: "Excellent" };
    if (value >= 50) return { text: "text-warning", bg: "bg-warning/10 border-warning/30", label: "Fair" };
    return { text: "text-destructive", bg: "bg-destructive/10 border-destructive/30", label: "Poor" };
  };

  const overall = getColor(score.overall);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="rounded-lg border border-border bg-card p-4 space-y-3"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Quality Score</h3>
        </div>
        <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${overall.bg} ${overall.text}`}>
          {overall.label}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-md bg-secondary p-2.5 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider">
            <Shield className="w-3 h-3" /> Overall
          </div>
          <p className={`text-lg font-bold font-mono ${overall.text}`}>{score.overall}</p>
        </div>

        <div className="rounded-md bg-secondary p-2.5 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider">
            <Eye className="w-3 h-3" /> Face Conf.
          </div>
          <p className={`text-lg font-bold font-mono ${getColor(score.face_confidence).text}`}>
            {score.face_confidence}
          </p>
        </div>

        <div className="rounded-md bg-secondary p-2.5 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider">
            <Scan className="w-3 h-3" /> Sharpness
          </div>
          <p className={`text-lg font-bold font-mono ${getColor(score.sharpness).text}`}>{score.sharpness}</p>
        </div>

        <div className="rounded-md bg-secondary p-2.5 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider">
            <AlertTriangle className="w-3 h-3" /> Issues
          </div>
          <div className="flex gap-1.5 mt-0.5">
            {score.blur_detected && (
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-warning/15 text-warning border border-warning/30">
                BLUR
              </span>
            )}
            {score.artifacts && (
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-destructive/15 text-destructive border border-destructive/30">
                ARTIFACTS
              </span>
            )}
            {!score.blur_detected && !score.artifacts && (
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-success/15 text-success border border-success/30">
                CLEAN
              </span>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default QualityScoreBadge;
