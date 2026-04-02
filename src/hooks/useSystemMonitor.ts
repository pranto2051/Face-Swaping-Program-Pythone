import { useEffect, useRef } from "react";
import { useStudioStore } from "@/store/useStudioStore";

const API_BASE = "http://127.0.0.1:5001/api";

export function useSystemMonitor(intervalMs = 5000) {
  const setCpuStats = useStudioStore((s) => s.setCpuStats);
  const setRamStats = useStudioStore((s) => s.setRamStats);
  const timerRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [cpuRes, ramRes] = await Promise.allSettled([
          fetch(`${API_BASE}/cpu`),
          fetch(`${API_BASE}/ram`),
        ]);

        if (cpuRes.status === "fulfilled" && cpuRes.value.ok) {
          setCpuStats(await cpuRes.value.json());
        }
        if (ramRes.status === "fulfilled" && ramRes.value.ok) {
          setRamStats(await ramRes.value.json());
        }
      } catch {
        // Endpoints may not exist yet
      }
    };

    fetchStats();
    timerRef.current = setInterval(fetchStats, intervalMs);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [intervalMs, setCpuStats, setRamStats]);
}
