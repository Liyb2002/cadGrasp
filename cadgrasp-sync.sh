#!/bin/bash
set -euo pipefail

LOCAL_DIR='/Users/yuanboli/Documents/GitHub/cadGrasp'
REMOTE_HOST='yli581@100.90.47.96'
REMOTE_DIR='/home/yli581/Desktop/cadGrasp'
LOCAL_BACKUPS='/Users/yuanboli/Desktop/cadGrasp-sync-backups'
REMOTE_BACKUPS='/home/yli581/.local/share/cadgrasp-sync/backups'

usage() {
    cat <<'EOF'
在 Mac 本地终端运行：
  bash /Users/yuanboli/Documents/GitHub/cadGrasp/cadgrasp-sync.sh push              上传：Mac → Ubuntu
  bash /Users/yuanboli/Documents/GitHub/cadGrasp/cadgrasp-sync.sh pull              下载：Ubuntu → Mac
  bash /Users/yuanboli/Documents/GitHub/cadGrasp/cadgrasp-sync.sh push --dry-run    预览上传
  bash /Users/yuanboli/Documents/GitHub/cadGrasp/cadgrasp-sync.sh pull --dry-run    预览下载
  bash /Users/yuanboli/Documents/GitHub/cadGrasp/cadgrasp-sync.sh shell             登录 Ubuntu 项目目录

只传输新增或发生变化的文件；保留目标端多出的文件。
同名文件以传输来源为准，被覆盖的旧文件另存到目标端的备份目录。
包含 objects、output 和其他被 Git 忽略的数据，不读取 .gitignore。
不传输 .git、.venv、__pycache__、.pytest_cache、.DS_Store 和传输临时目录。
Python/Conda/CUDA 环境需要在 Ubuntu 上单独准备。
EOF
}

if [[ $# -eq 0 || ${1:-} == '--help' || ${1:-} == '-h' ]]; then
    usage
    exit 0
fi

mode=$1
shift
dry_run=false
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 1 || $1 != '--dry-run' || $mode == 'shell' ]]; then
        usage >&2
        exit 2
    fi
    dry_run=true
fi

case "$mode" in
    shell)
        exec ssh -t -o ConnectTimeout=15 -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 "$REMOTE_HOST" \
            "cd '$REMOTE_DIR' && exec bash -l"
        ;;
    push|pull) ;;
    *) usage >&2; exit 2 ;;
esac

if [[ ! -d $LOCAL_DIR ]]; then
    printf '找不到 Mac 项目目录：%s\n' "$LOCAL_DIR" >&2
    exit 1
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
rsync_args=(
    -rltp
    --partial-dir=.rsync-partial
    --progress
    --itemize-changes
    --stats
    --backup
    --exclude=.git
    --exclude=.venv/
    --exclude=__pycache__/
    --exclude=.pytest_cache/
    --exclude=.DS_Store
    --exclude=.rsync-partial/
    -e 'ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3'
)

if [[ $mode == 'push' ]]; then
    source_path="$LOCAL_DIR/"
    destination_path="$REMOTE_HOST:$REMOTE_DIR/"
    backup_dir="$REMOTE_BACKUPS/$stamp"
else
    source_path="$REMOTE_HOST:$REMOTE_DIR/"
    destination_path="$LOCAL_DIR/"
    backup_dir="$LOCAL_BACKUPS/$stamp"
fi

printf '来源：%s\n目标：%s\n旧文件备份（在接收端）：%s\n' \
    "$source_path" "$destination_path" "$backup_dir"
if [[ $dry_run == true ]]; then
    printf '模式：仅预览文件变化\n'
    rsync_args+=(--dry-run)
fi

rsync "${rsync_args[@]}" --backup-dir="$backup_dir" \
    "$source_path" "$destination_path"

if [[ $dry_run == false ]]; then
    printf '\n同步完成。\n'
fi
