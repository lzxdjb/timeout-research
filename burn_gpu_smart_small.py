import torch
import torch.multiprocessing as mp
import signal
import sys
import os
import time
import logging
from pathlib import Path

logging.disable(logging.CRITICAL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GPU%(gpu_id)s] %(message)s",
    datefmt="%H:%M:%S"
)

PROJECT_ROOT = Path(__file__).resolve().parent
PID_LOG_DIR = PROJECT_ROOT / "formal_run_log"
PID_FILE = PID_LOG_DIR / "burn_gpu_smart.pid"
CHILD_PID_FILE = PID_LOG_DIR / "burn_gpu_smart.child_pids"
LEGACY_CHILD_PID_FILE = Path("/tmp/burn_gpu.pids")


def write_pid_files(parent_pid, child_pids):
    PID_LOG_DIR.mkdir(parents=True, exist_ok=True)

    PID_FILE.write_text(f"{parent_pid}\n")

    child_payload = "\n".join(str(pid) for pid in child_pids)
    if child_payload:
        child_payload += "\n"

    CHILD_PID_FILE.write_text(child_payload)
    LEGACY_CHILD_PID_FILE.write_text(child_payload)

def stress_gpu(gpu_id):
    # Detach from parent process group so Ray's cleanup doesn't kill us
    os.setsid()

    # Handle signals gracefully in child
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    log = logging.LoggerAdapter(logging.getLogger(), {"gpu_id": gpu_id})

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    log.info(f"Starting GPU stress on GPU {gpu_id}")

    size = 400
    iteration = 0

    while True:
        try:
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            iteration += 1

            if iteration % 1000 == 0:
                log.info(f"GPU {gpu_id} - iteration {iteration}")

        except torch.cuda.OutOfMemoryError:
            log.warning(f"GPU {gpu_id} OOM — reducing matrix size and retrying")
            size = max(64, size // 2)
            torch.cuda.empty_cache()
            time.sleep(1)

        except Exception as e:
            log.error(f"GPU {gpu_id} error: {e}, restarting loop in 2s")
            time.sleep(2)


def main():
    if not torch.cuda.is_available():
        return

    gpu_count = torch.cuda.device_count()

    processes = []
    for gpu_id in range(gpu_count):
        p = mp.Process(target=stress_gpu, args=(gpu_id,), daemon=False)
        p.start()
        processes.append(p)

    # Save the parent PID plus child PIDs for later cleanup.
    write_pid_files(os.getpid(), [p.pid for p in processes if p.pid is not None])

    # Parent handles signals but doesn't propagate to children (they're in own session)
    def shutdown(sig, frame):
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
