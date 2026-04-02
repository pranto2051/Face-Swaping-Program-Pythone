import React, { useCallback, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useStudioStore } from "@/store/useStudioStore";
import { Scan, Check } from "lucide-react";

interface FaceSelectorOverlayProps {
  imageUrl: string;
  loading?: boolean;
  locked?: boolean;
}

const FaceSelectorOverlay: React.FC<FaceSelectorOverlayProps> = ({
  imageUrl,
  loading = false,
  locked = false,
}) => {
  const { detectedFaces, faceAssignments, toggleFaceSelection } = useStudioStore();
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  const isFaceSelected = useCallback(
    (faceId: number) => faceAssignments.some((a) => a.targetFaceId === faceId),
    [faceAssignments]
  );

  const selectedFaceId = useMemo(
    () => faceAssignments[0]?.targetFaceId,
    [faceAssignments]
  );

  if ((detectedFaces.length <= 1 && !loading) || (!imageUrl && !loading)) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-3"
    >
      <div className="flex items-center gap-2 text-sm">
        <Scan className="w-4 h-4 text-primary" />
        <span className="font-medium text-foreground">
          {loading ? "Detecting faces..." : "Select a face to continue"}
        </span>
      </div>

      <div className="rounded-lg border border-border bg-secondary p-2">
        <div className="relative inline-block max-w-full">
          <img
            src={imageUrl}
            alt="Target"
            className="block max-h-80 w-auto max-w-full object-contain rounded"
            onLoad={(e) => {
              const img = e.currentTarget;
              setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
            }}
          />

          {/* Face bounding box overlays */}
          {naturalSize &&
            detectedFaces.map((face) => {
              const selected = isFaceSelected(face.id);
              const [x1, y1, x2, y2] = face.bbox;
              const left = (x1 / naturalSize.width) * 100;
              const top = (y1 / naturalSize.height) * 100;
              const width = ((x2 - x1) / naturalSize.width) * 100;
              const height = ((y2 - y1) / naturalSize.height) * 100;

              return (
                <motion.button
                  key={face.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  onClick={() => !locked && toggleFaceSelection(face.id)}
                  disabled={locked}
                  style={{
                    position: "absolute",
                    left: `${left}%`,
                    top: `${top}%`,
                    width: `${width}%`,
                    height: `${height}%`,
                  }}
                  className={`border-2 rounded transition-all ${
                    locked ? "cursor-not-allowed opacity-80" : "cursor-pointer"
                  } ${
                    selected
                      ? "border-blue-500 bg-blue-500/15 shadow-[0_0_12px_rgba(59,130,246,0.35)]"
                      : "border-muted-foreground/50 hover:border-blue-400/80 hover:bg-blue-500/10 bg-transparent"
                  }`}
                  title={`Face ${face.id + 1}`}
                >
                  <span
                    className={`absolute -top-2.5 -left-2.5 px-1.5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center ${
                      selected
                        ? "bg-blue-500 text-white"
                        : "bg-secondary text-muted-foreground border border-border"
                    }`}
                  >
                    Face {face.id + 1}
                  </span>
                  {selected && (
                    <Check className="absolute -top-2 -right-2 w-4 h-4 text-blue-500" />
                  )}
                  <span className="absolute bottom-0 left-0 right-0 text-[9px] font-mono text-center bg-background/70 text-muted-foreground py-0.5">
                    {(face.confidence * 100).toFixed(0)}%
                  </span>
                </motion.button>
              );
            })}
        </div>
      </div>

      {detectedFaces.length > 0 && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground font-mono">
            {selectedFaceId !== undefined ? `Selected: Face ${selectedFaceId + 1}` : "No face selected"}
          </span>
          <span className="text-xs text-muted-foreground">
            Click a box to select
          </span>
        </div>
      )}
    </motion.div>
  );
};

export default FaceSelectorOverlay;
