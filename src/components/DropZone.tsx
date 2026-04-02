import React, { useCallback, useState, useRef, useEffect, useMemo } from "react";
import { Upload, X, Image as ImageIcon, RefreshCw, Loader2, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ScrollArea } from "@/components/ui/scroll-area";

interface DropZoneProps {
  label: string;
  accept?: string;
  multiple?: boolean;
  files: File[];
  onFiles: (files: File[]) => void;
  maxFiles?: number;
  compact?: boolean;
}

const DropZone: React.FC<DropZoneProps> = ({
  label,
  accept = "image/*",
  multiple = false,
  files,
  onFiles,
  maxFiles = 1,
  compact = false,
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState(false);

  // Create and manage blob URLs with proper cleanup
  const previewUrls = useMemo(() => {
    return files.map((file) => URL.createObjectURL(file));
  }, [files]);

  // Cleanup blob URLs when files change or component unmounts
  useEffect(() => {
    return () => {
      previewUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [previewUrls]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = Array.from(e.dataTransfer.files).slice(0, maxFiles);
      onFiles(multiple ? [...files, ...dropped].slice(0, maxFiles) : dropped.slice(0, 1));
    },
    [files, maxFiles, multiple, onFiles]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files) return;
      const selected = Array.from(e.target.files).slice(0, maxFiles);
      onFiles(multiple ? [...files, ...selected].slice(0, maxFiles) : selected.slice(0, 1));
    },
    [files, maxFiles, multiple, onFiles]
  );

  const replaceInputRef = useRef<HTMLInputElement>(null);
  const [replaceIndex, setReplaceIndex] = useState<number | null>(null);

  const removeFile = (index: number) => {
    onFiles(files.filter((_, i) => i !== index));
  };

  const handleReplace = (index: number) => {
    setReplaceIndex(index);
    replaceInputRef.current?.click();
  };

  const handleReplaceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || replaceIndex === null) return;
    const newFile = e.target.files[0];
    const updated = [...files];
    updated[replaceIndex] = newFile;
    onFiles(updated);
    setReplaceIndex(null);
    e.target.value = "";
  };

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground">{label}</label>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`relative rounded-lg border-2 border-dashed transition-all duration-300 cursor-pointer ${
          dragOver ? "drop-zone-active border-primary" : "border-border hover:border-muted-foreground"
        } ${compact ? "p-4" : "p-8"}`}
      >
        <input
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />

        {files.length === 0 ? (
          <div className="flex flex-col items-center gap-3 text-muted-foreground">
            <div className="w-12 h-12 rounded-full bg-secondary flex items-center justify-center">
              <Upload className="w-5 h-5" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium">Drop files here or click to browse</p>
              <p className="text-xs mt-1">{multiple ? `Up to ${maxFiles} files` : "Single file"}</p>
            </div>
          </div>
        ) : multiple ? (
          <ScrollArea className="max-h-72 w-full">
            <div className="grid grid-cols-4 gap-2 w-full p-1">
              <AnimatePresence>
                {files.map((file, i) => (
                  <motion.div
                    key={file.name + i}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="relative group aspect-square"
                  >
                    <div className="w-full h-full rounded-md overflow-hidden bg-secondary border border-border">
                      {file.type.startsWith("image/") ? (
                        <img
                          src={previewUrls[i]}
                          alt="Preview"
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            console.error("Image preview error:", file.name);
                            e.currentTarget.style.display = "none";
                          }}
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <ImageIcon className="w-6 h-6 text-muted-foreground" />
                        </div>
                      )}
                    </div>
                    <div className="absolute top-1 right-1 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={(e) => { e.stopPropagation(); handleReplace(i); }}
                        className="w-5 h-5 rounded-full bg-secondary/90 border border-border flex items-center justify-center"
                      >
                        <RefreshCw className="w-2.5 h-2.5 text-foreground" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                        className="w-5 h-5 rounded-full bg-destructive flex items-center justify-center"
                      >
                        <X className="w-2.5 h-2.5 text-destructive-foreground" />
                      </button>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </ScrollArea>
        ) : (
          <div className="w-full">
            <AnimatePresence>
              {files.map((file, i) => (
                <motion.div
                  key={file.name + i}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="relative group w-full"
                >
                  <div className="w-full rounded-md overflow-hidden bg-secondary border border-border">
                    {file.type.startsWith("image/") ? (
                      <img
                        src={previewUrls[i]}
                        alt="Preview"
                        className="w-full max-h-64 object-contain"
                        onError={(e) => {
                          console.error("Image preview error:", file.name);
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    ) : file.type.startsWith("video/") ? (
                      <div className="relative">
                        {videoLoading && (
                          <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-md z-10">
                            <Loader2 className="w-8 h-8 text-white animate-spin" />
                          </div>
                        )}
                        {videoError ? (
                          <div className="w-full h-64 flex flex-col items-center justify-center gap-3 bg-destructive/10 rounded-md">
                            <AlertCircle className="w-12 h-12 text-destructive" />
                            <div className="text-center px-4">
                              <p className="text-sm font-medium text-destructive">Video playback error</p>
                              <p className="text-xs text-muted-foreground mt-1">
                                Format may not be supported. Try MP4, WebM, or OGG.
                              </p>
                            </div>
                          </div>
                        ) : (
                          <>
                            <video
                              src={previewUrls[i]}
                              className="w-full max-h-96 object-contain rounded-md bg-black"
                              controls
                              preload="metadata"
                              onLoadStart={() => {
                                setVideoLoading(true);
                                setVideoError(false);
                              }}
                              onLoadedData={() => setVideoLoading(false)}
                              onError={(e) => {
                                console.error("Video preview error:", file.name, e);
                                setVideoLoading(false);
                                setVideoError(true);
                              }}
                            />
                            <div className="absolute bottom-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded text-[10px] font-mono text-white">
                              {file.name} ({(file.size / (1024 * 1024)).toFixed(1)} MB)
                            </div>
                          </>
                        )}
                      </div>
                    ) : (
                      <div className="w-full h-32 flex items-center justify-center">
                        <ImageIcon className="w-8 h-8 text-muted-foreground" />
                      </div>
                    )}
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                    className="absolute top-2 right-2 w-6 h-6 rounded-full bg-destructive flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="w-3 h-3 text-destructive-foreground" />
                  </button>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
      <input
        type="file"
        ref={replaceInputRef}
        accept={accept}
        onChange={handleReplaceChange}
        className="hidden"
      />
    </div>
  );
};

export default DropZone;
