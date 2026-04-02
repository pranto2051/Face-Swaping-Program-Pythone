import React, { useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, AlertCircle, Download, Expand } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, type BatchResultItem } from "@/services/api";
import { toast } from "sonner";

interface ResultCardProps {
  item: BatchResultItem;
  index?: number;
  onClick?: () => void;
}

const ResultCard: React.FC<ResultCardProps> = ({ item, index = 0, onClick }) => {
  const [downloading, setDownloading] = useState(false);
  const isError = item.status === "error";

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!item.output_url) return;
    
    setDownloading(true);
    try {
      await api.downloadFile(item.output_url, item.filename);
      toast.success("Download Successful ✅", {
        description: `${item.filename} has been saved.`,
        duration: 3000,
      });
    } catch (err) {
      toast.error("Download Failed ❌");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08 }}
      className="rounded-lg border border-border bg-card overflow-hidden group"
    >
      <div
        className="aspect-square relative bg-secondary cursor-pointer"
        onClick={item.output_url ? onClick : undefined}
      >
        {item.output_url ? (
          <>
            <img src={item.output_url} alt="Result" className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-background/0 group-hover:bg-background/40 transition-colors flex items-center justify-center">
              <Expand className="w-6 h-6 text-foreground opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
            </div>
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-destructive" />
          </div>
        )}
      </div>

      <div className="p-3 flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          {isError ? (
            <AlertCircle className="w-4 h-4 text-destructive shrink-0" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-success shrink-0" />
          )}
          <span className="text-xs text-muted-foreground truncate">{item.filename}</span>
        </div>
        {item.output_url && (
          <Button 
            variant="ghost" 
            size="sm" 
            className="h-7 px-2 shrink-0" 
            disabled={downloading}
            onClick={handleDownload}
          >
            {downloading ? (
              <div className="w-3.5 h-3.5 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
          </Button>
        )}
      </div>

      {item.message && (
        <div className="px-3 pb-3">
          <p className="text-xs text-destructive">{item.message}</p>
        </div>
      )}
    </motion.div>
  );
};

export default ResultCard;
