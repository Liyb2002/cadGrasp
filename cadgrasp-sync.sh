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
同步 codes、objects、simulation、slides 和根目录的非隐藏文件、.gitignore。
保留这些目录内的模型、数据和算法输出；其他顶层目录不参与同步。
不传输 Git 元数据、虚拟环境、node_modules、工具缓存和传输临时目录。
下载时保留本机 cadgrasp-sync.sh，避免远端旧脚本覆盖同步工具。
正式传输前自动预览文件数和待传输数量；预览与传输复用 SSH 连接。
需要 Homebrew rsync 3.1+（brew install rsync）；连续 300 秒无传输数据会退出。
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

# macOS 自带的 openrsync 与标准 rsync 不同；明确选择 Homebrew 版本。
if [[ -x /opt/homebrew/bin/rsync ]]; then
    rsync_bin=/opt/homebrew/bin/rsync
elif [[ -x /usr/local/bin/rsync ]]; then
    rsync_bin=/usr/local/bin/rsync
else
    printf '请先安装标准 rsync：brew install rsync\n' >&2
    exit 1
fi
rsync_version=$("$rsync_bin" --version)
if [[ $rsync_version =~ ^rsync[[:space:]]+version[[:space:]]+([0-9]+)\.([0-9]+) ]] &&
    (( BASH_REMATCH[1] > 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] >= 1) )); then
    printf '同步工具：%s（%s）\n' "$rsync_bin" "${rsync_version%%$'\n'*}"
else
    printf '需要标准 rsync 3.1+，请运行：brew upgrade rsync\n' >&2
    exit 1
fi

run_dir=$(mktemp -d /tmp/cadgrasp-sync.XXXXXX)
control_socket="$run_dir/ssh"
cleanup() {
    if [[ -S $control_socket ]]; then
        ssh -S "$control_socket" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
    fi
    rm -rf -- "$run_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
rsync_args=(
    -rltp
    --partial-dir=.rsync-partial
    --info=progress2,name0
    --human-readable
    --outbuf=L
    --timeout=300
    --stats
    --backup
    --exclude=.git
    '--exclude=.venv*/'
    --exclude=venv/
    '--exclude=venv-*/'
    '--exclude=venv_*/'
    --exclude=env/
    '--exclude=env-*/'
    '--exclude=env_*/'
    --exclude=node_modules/
    --exclude=.cache/
    --exclude=.conda/
    --exclude=.mamba/
    --exclude=__pycache__/
    --exclude=.pytest_cache/
    --exclude=.mypy_cache/
    --exclude=.ruff_cache/
    --exclude=.tox/
    --exclude=.nox/
    --exclude=.ipynb_checkpoints/
    '--exclude=*.pyc'
    '--exclude=*.pyo'
    --exclude=/.agents/
    --exclude=/.codex/
    --exclude=/.aws/
    --exclude=.DS_Store
    '--exclude=._*'
    --exclude=.rsync-partial/
    # 依赖/缓存排除规则必须在目录白名单之前，保证嵌套缓存也会跳过。
    --include=/.gitignore
    '--exclude=/.*'
    --include=/codes/
    --include=/objects/
    --include=/simulation/
    --include=/slides/
    '--exclude=/*/'
    -e "ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPersist=60 -o ControlPath=$control_socket"
)

if [[ $mode == 'push' ]]; then
    source_path="$LOCAL_DIR/"
    destination_path="$REMOTE_HOST:$REMOTE_DIR/"
    backup_dir="$REMOTE_BACKUPS/$stamp"
else
    source_path="$REMOTE_HOST:$REMOTE_DIR/"
    destination_path="$LOCAL_DIR/"
    backup_dir="$LOCAL_BACKUPS/$stamp"
    rsync_args+=(--exclude=/cadgrasp-sync.sh)
fi

printf '来源：%s\n目标：%s\n旧文件备份（在接收端）：%s\n' \
    "$source_path" "$destination_path" "$backup_dir"
printf '范围：codes/、objects/、simulation/、slides/、根目录项目文件；跳过依赖和缓存。\n'

run_sync() {
    local status
    if LC_ALL=C "$rsync_bin" "${rsync_args[@]}" --backup-dir="$backup_dir" \
        "$@" "$source_path" "$destination_path"; then
        return 0
    else
        status=$?
        printf '\n同步未完成（退出码 %s）。已完成的文件会在下次运行时跳过；可重新运行相同命令。\n' "$status" >&2
        return "$status"
    fi
}

if [[ $dry_run == true ]]; then
    printf '模式：仅预览文件变化\n'
    run_sync --dry-run --itemize-changes --info=progress0,name1
    exit 0
fi

printf '\n正在核对源端清单和待传输数量（不写入目标文件）…\n'
preview_file="$run_dir/preview.txt"
if run_sync --dry-run --info=progress0,name0 >"$preview_file"; then
    :
else
    status=$?
    cat "$preview_file" >&2
    exit "$status"
fi
awk '
    /^Number of files:/ {
        sub(/^[^:]*: /, ""); gsub(/reg:/, "文件:"); gsub(/dir:/, "目录:"); gsub(/link:/, "链接:")
        print "源端清单（包含目录）：" $0
    }
    /^Number of regular files transferred:/ { sub(/^[^:]*: /, ""); print "本次需要传输的文件：" $0 }
    /^Total transferred file size:/ { sub(/^[^:]*: /, ""); print "待传输文件总大小：" $0 }
' "$preview_file"

mkdir -p "$LOCAL_BACKUPS/logs"
log_file="$LOCAL_BACKUPS/logs/$stamp-$mode.log"
rsync_args+=(--log-file="$log_file")
printf '同步日志：%s\n\n' "$log_file"
run_sync
printf '\n同步完成。\n'
