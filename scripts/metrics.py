import torch
import wandb
class MetricsCollector:
    """Collect and calculate performance metrics."""
    def __init__(self, cuda_events:bool= True, wandb):
        self.cuda_events = cuda_events if torch.cuda.is_available() else False
        self.start_event = torch.cuda.Event(enable_timing=True) if self.cuda_events else None
        self.end_event = torch.cuda.Event(enable_timing=True) if self.cuda_events else None
        self.latencies = []
        self.gpu_memories = []
        self.cpu_memories = []
        self.gpu_utils = []
        self.cpu_utils = []
         
        self.wandb = wandb.init(
            project="InferenceMetrics",
            config={
                "cuda_events": self.cuda_events,
                "start_time":self.start_event,
                "end_event":self.end_event,
                "latencies": self.latencies,
                "gpu_memories": self.gpu_memories,
                "cpu_memories": self.cpu_memories,
                "gpu_utils": self.gpu_utils,
                "cpu_utils": self.cpu_utils,
            }
        )
    def run_inference(self):
        self.start_event.record() if self.cuda_events else None