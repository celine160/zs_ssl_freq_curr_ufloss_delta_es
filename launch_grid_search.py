import argparse
import subprocess
import time
import queue
import threading
import os

def run_job(gpu_id, lambda_val):
    print(f"[+] Starting job with lambda_uf={lambda_val} on GPU {gpu_id}")
    
    # You can edit these baseline arguments if you want to test other configurations!
    cmd = [
        "python3", "recon_code.py",
        "--data_opt", "AXFLAIR",
        "--data_dir", "data/processed_mat_201_6002867/file_brain_AXFLAIR_201_6002867_slice000_R4_ACS24.mat",
        "--lambda_uf", str(lambda_val)
    ]
    
    # We pass the current environment but override CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # We route stdout to DEVNULL so the console doesn't become a mess of 9 different trainings.
    # The progress will be safely logged to 'training_log.txt' inside each model's directory!
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=None
    )
    process.wait()
    print(f"[-] Job with lambda_uf={lambda_val} on GPU {gpu_id} finished.")
    return gpu_id

def main():
    parser = argparse.ArgumentParser(description="Grid Search Launcher for ZS-SSL")
    parser.add_argument('--lambdas', nargs='+', type=float, required=True, help="List of lambda_uf values (e.g. 0.005 0.01 0.05)")
    parser.add_argument('--gpus', nargs='+', type=str, required=True, help="List of GPU IDs to use (e.g. 0 1 2)")
    args = parser.parse_args()

    print(f"Starting Grid Search!")
    print(f"Testing {len(args.lambdas)} lambda values: {args.lambdas}")
    print(f"Using {len(args.gpus)} GPUs: {args.gpus}\n")

    # Thread-safe queue to manage free GPUs
    gpu_queue = queue.Queue()
    for g in args.gpus:
        gpu_queue.put(g)

    def worker(lambda_val):
        # This will block until a GPU is placed into the queue (i.e. is free)
        gpu_id = gpu_queue.get()
        run_job(gpu_id, lambda_val)
        # When finished, put the GPU back in the queue for the next job
        gpu_queue.put(gpu_id)

    threads = []
    for val in args.lambdas:
        t = threading.Thread(target=worker, args=(val,))
        t.start()
        threads.append(t)
        # Small 3 second delay between launches so PyTorch/CUDA initializations don't collide
        time.sleep(3)

    for t in threads:
        t.join()

    print("\nAll grid search jobs completed successfully!")

if __name__ == "__main__":
    main()
