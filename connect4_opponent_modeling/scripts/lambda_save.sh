#!/bin/bash
# Save experiment artifacts from Lambda instance to local machine.
# Run this LOCALLY (not on Lambda) before terminating the instance.
#
# Usage:
#   bash scripts/lambda_save.sh 192.222.51.110
#   bash scripts/lambda_save.sh <instance-ip>

set -e

IP=${1:?Usage: bash scripts/lambda_save.sh <instance-ip>}
REMOTE="ubuntu@$IP"
LOCAL_BACKUP="./lambda_backup_$(date +%Y%m%d_%H%M)"
PROJECT="~/wikipedia-agent/connect4_opponent_modeling"

echo "=== Saving Lambda Instance Artifacts ==="
echo "  Instance: $IP"
echo "  Local backup: $LOCAL_BACKUP"
echo ""

mkdir -p "$LOCAL_BACKUP"

# Save checkpoints
echo "Downloading checkpoints..."
rsync -avz --progress "$REMOTE:$PROJECT/checkpoints/" "$LOCAL_BACKUP/checkpoints/" 2>&1 | tail -3
echo ""

# Save training logs
echo "Downloading logs..."
rsync -avz --progress "$REMOTE:$PROJECT/logs/" "$LOCAL_BACKUP/logs/" 2>&1 | tail -3
echo ""

# Save results
echo "Downloading results..."
rsync -avz --progress "$REMOTE:$PROJECT/results/" "$LOCAL_BACKUP/results/" 2>&1 | tail -3 || echo "  No results yet"
echo ""

# Save data (probe positions, sft data)
echo "Downloading data..."
rsync -avz --progress "$REMOTE:$PROJECT/data/" "$LOCAL_BACKUP/data/" 2>&1 | tail -3
echo ""

echo "============================================"
echo "Backup complete: $LOCAL_BACKUP"
du -sh "$LOCAL_BACKUP"
echo ""
echo "To restore on a new instance:"
echo "  1. bash scripts/lambda_resume.sh"
echo "  2. bash scripts/lambda_restore.sh <new-instance-ip>"
echo "============================================"
