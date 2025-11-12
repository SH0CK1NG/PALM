# 查看并杀掉 lijialo+ 用户在 GPU 0-3 上的全部进程
for gpu in 0 1 2 3; do
  # 查出该 GPU 上由 lijialo+ 启动的进程 PID
  pids=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader | xargs -r ps -u lijialong -o pid= --no-headers | xargs)
  # 杀掉这些进程
  if [ -n "$pids" ]; then
    echo "Killing processes on GPU $gpu: $pids"
    kill -9 $pids
  fi
done
