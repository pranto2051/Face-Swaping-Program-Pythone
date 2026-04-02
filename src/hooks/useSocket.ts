import { useEffect, useRef } from "react";
import { io, Socket } from "socket.io-client";
import { useStudioStore } from "@/store/useStudioStore";
import type { ProgressData } from "@/store/types";
import { outputUrl } from "@/services/api";
import { toast } from "sonner";

const SOCKET_URL = "http://127.0.0.1:5001";

export function useSocket() {
  const socketRef = useRef<Socket | null>(null);
  const { 
    setProgress, 
    setSocketConnected, 
    updateJobStatus, 
    setQualityScore,
    addJobResult,
    setProcessing,
    setBatchZipUrl,
  } = useStudioStore();

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionDelay: 3000,
    });
    socketRef.current = socket;

    socket.on("connect", () => {
      setSocketConnected(true);
    });

    socket.on("disconnect", () => {
      setSocketConnected(false);
    });

    socket.on("progress", (data: ProgressData) => {
      setProgress(data.job_id, data);
      if (data.status === "cancelled") {
        updateJobStatus(data.job_id, "cancelled");
        setProcessing(false);
      } else {
        updateJobStatus(data.job_id, "processing");
      }
    });

    socket.on("complete", (data: { job_id: string; output_url?: string | any[]; quality_score?: any; zip_url?: string | null }) => {
      updateJobStatus(data.job_id, "completed");
      setProcessing(false);
      
      // Show success notification
      toast.success("Processing Complete! 🎉", {
        description: "Your result is ready to download.",
        duration: 5000,
      });
      
      if (data.output_url) {
        // Handle batch results (array)
        if (Array.isArray(data.output_url)) {
          const processedResults = data.output_url.map((r: any) => ({
            filename: r.filename,
            output_url: outputUrl(r.output_url),
          }));
          addJobResult(data.job_id, processedResults);
        } else {
          // Handle single result (string)
          addJobResult(data.job_id, outputUrl(data.output_url));
        }
      }
      if (data.quality_score) {
        setQualityScore(data.quality_score);
      }
      if (data.zip_url) {
        setBatchZipUrl(outputUrl(data.zip_url));
      }
    });

    socket.on("error", (data: { job_id: string; message: string }) => {
      updateJobStatus(data.job_id, "failed");
      setProcessing(false);
      
      // Show error notification
      toast.error("Processing Failed", {
        description: data.message || "An error occurred during processing.",
        duration: 5000,
      });
    });

    return () => {
      socket.disconnect();
    };
  }, [setProgress, setSocketConnected, updateJobStatus, setQualityScore]);

  return socketRef;
}
