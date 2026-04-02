import React, { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, ChevronLeft, ChevronRight, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/services/api";
import { toast } from "sonner";

interface LightboxItem {
  url: string;
  filename: string;
}

interface ImageLightboxProps {
  items: LightboxItem[];
  currentIndex: number;
  open: boolean;
  onClose: () => void;
  onIndexChange: (index: number) => void;
}

const ImageLightbox: React.FC<ImageLightboxProps> = ({
  items,
  currentIndex,
  open,
  onClose,
  onIndexChange,
}) => {
  const [isDownloading, setIsDownloading] = useState(false);
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex < items.length - 1;
  const current = items[currentIndex];

  const goPrev = useCallback(() => {
    if (hasPrev) onIndexChange(currentIndex - 1);
  }, [hasPrev, currentIndex, onIndexChange]);

  const goNext = useCallback(() => {
    if (hasNext) onIndexChange(currentIndex + 1);
  }, [hasNext, currentIndex, onIndexChange]);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") goPrev();
      if (e.key === "ArrowRight") goNext();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, onClose, goPrev, goNext]);

  const handleDownload = useCallback(async () => {
    if (!current) return;
    setIsDownloading(true);
    try {
      await api.downloadFile(current.url, current.filename);
      toast.success("Download Successful ✅", {
        description: `${current.filename} has been saved to your computer.`,
        duration: 4000,
      });
    } catch (error: any) {
      toast.error("Download Failed ❌", {
        description: error?.message || "File not found or inaccessible.",
        duration: 4000,
      });
    } finally {
      setIsDownloading(false);
    }
  }, [current]);

  if (!current) return null;

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[100] flex items-center justify-center"
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-background/90 backdrop-blur-md"
            onClick={onClose}
          />

          {/* Top bar */}
          <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4">
            <span className="text-sm font-mono text-muted-foreground">
              {currentIndex + 1} / {items.length}
              <span className="ml-3 text-xs text-foreground/60 truncate max-w-[200px] inline-block align-bottom">
                {current.filename}
              </span>
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                className="gap-2 border-border bg-card/80 backdrop-blur-sm"
                onClick={handleDownload}
                disabled={isDownloading}
              >
                {isDownloading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" />
                    Download
                  </>
                )}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-9 w-9 text-foreground hover:bg-destructive/10"
              >
                <X className="w-5 h-5" />
              </Button>
            </div>
          </div>

          {/* Image */}
          <motion.div
            key={currentIndex}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="relative z-10 max-w-[90vw] max-h-[80vh] flex items-center justify-center"
          >
            <img
              src={current.url}
              alt={current.filename}
              className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-2xl"
            />
          </motion.div>

          {/* Left Arrow */}
          {hasPrev && (
            <button
              onClick={goPrev}
              className="absolute left-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full bg-card/80 backdrop-blur-sm border border-border text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
            >
              <ChevronLeft className="w-6 h-6" />
            </button>
          )}

          {/* Right Arrow */}
          {hasNext && (
            <button
              onClick={goNext}
              className="absolute right-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full bg-card/80 backdrop-blur-sm border border-border text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
            >
              <ChevronRight className="w-6 h-6" />
            </button>
          )}

          {/* Bottom thumbnail strip */}
          {items.length > 1 && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 p-2 rounded-lg bg-card/80 backdrop-blur-sm border border-border max-w-[90vw] overflow-x-auto">
              {items.map((item, i) => (
                <button
                  key={i}
                  onClick={() => onIndexChange(i)}
                  className={`shrink-0 w-12 h-12 rounded-md overflow-hidden border-2 transition-all ${
                    i === currentIndex
                      ? "border-primary ring-1 ring-primary/30 scale-110"
                      : "border-transparent opacity-50 hover:opacity-80"
                  }`}
                >
                  <img
                    src={item.url}
                    alt={item.filename}
                    className="w-full h-full object-cover"
                  />
                </button>
              ))}
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ImageLightbox;
