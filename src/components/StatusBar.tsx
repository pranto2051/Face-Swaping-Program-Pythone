import React, { useEffect, useState } from "react";
import { Wifi, WifiOff, Cpu, CheckCircle2, XCircle, RefreshCw, Zap } from "lucide-react";
import { api, type HealthStatus } from "@/services/api";
import { useStudioStore } from "@/store/useStudioStore";

const StatusBar: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState(false);
  const [checking, setChecking] = useState(false);
  const { socketConnected, jobs } = useStudioStore();
  
  const activeJobs = jobs.filter(j => j.status === "processing" || j.status === "queued").length;

  const checkHealth = async () => {
    setChecking(true);
    try {
      const data = await api.health();
      setHealth(data);
      setError(false);
    } catch {
      setHealth(null);
      setError(true);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const connected = !!health && !error;

  return (
    <div className="flex items-center gap-4 text-xs font-mono flex-wrap justify-center">
      <button onClick={checkHealth} className="flex items-center gap-1.5 hover:opacity-80 transition-opacity">
        {checking ? (
          <RefreshCw className="w-3.5 h-3.5 text-muted-foreground animate-spin" />
        ) : connected ? (
          <Wifi className="w-3.5 h-3.5 text-success" />
        ) : (
          <WifiOff className="w-3.5 h-3.5 text-destructive" />
        )}
        <span className={connected ? "text-success" : "text-destructive"}>
          {connected ? "Backend" : "Offline"}
        </span>
      </button>
      
      {/* WebSocket Status */}
      <div className="flex items-center gap-1.5">
        <div className={`w-2 h-2 rounded-full ${socketConnected ? "bg-success animate-pulse" : "bg-muted-foreground/50"}`} />
        <span className={socketConnected ? "text-success" : "text-muted-foreground"}>
          {socketConnected ? "Live" : "Socket"}
        </span>
      </div>
      
      {/* Active Jobs Indicator */}
      {activeJobs > 0 && (
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-primary/10 border border-primary/30">
          <Zap className="w-3 h-3 text-primary animate-pulse" />
          <span className="text-primary font-semibold">{activeJobs} Active</span>
        </div>
      )}

      {connected && health && (
        <>
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-primary" />
            <span className="text-muted-foreground">
              {health.gpu}
              {health.cuda && <span className="text-primary ml-1">CUDA</span>}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            {health.models_loaded ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-success" />
            ) : (
              <XCircle className="w-3.5 h-3.5 text-warning" />
            )}
            <span className="text-muted-foreground">
              {health.models_loaded ? "Models Ready" : "Loading Models"}
            </span>
          </div>
        </>
      )}
    </div>
  );
};

export default StatusBar;
