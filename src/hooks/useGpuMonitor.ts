import { useEffect, useRef } from "react";
import { useStudioStore } from "@/store/useStudioStore";

const API_BASE = "http://127.0.0.1:5001/api";

export function useGpuMonitor(intervalMs = 5000) {
  const setGpuStats = useStudioStore((s) => s.setGpuStats);
  const timerRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    const fetchGpu = async () => {
      try {
        const res = await fetch(`${API_BASE}/gpu`);
        if (res.ok) {
          const data = await res.json();
          setGpuStats(data);
        }
      } catch {
        // GPU endpoint may not exist yet
      }
    };

    fetchGpu();
    timerRef.current = setInterval(fetchGpu, intervalMs);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [intervalMs, setGpuStats]);
}
