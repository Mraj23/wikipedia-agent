#!/bin/bash
# Restore experiment artifacts to a new Lambda instance.
# Run this LOCALLY after lambda_resume.sh has set up the new instance.
#
# Usage:
#   bash scripts/lambda_restore.sh <instance-ip> [backup-dir]
#
# If backup-dir is omitted, uses the most recent lambda_backup_* directory.

set -e

IP=${1:?Usage: bash scripts/lambda_restore.sh <instance-ip> [backup-dir]}
REMOTE="ubuntu@$IP"
PROJECT="~/wikipedia-agent/connect4_opponent_modeling"

# Find backup dir
if [ -n "$2" ]; then
    LOCAL_BACKUP="$2"
else
    LOCAL_BACKUP=$(ls -dt lambda_backup_* 2>/dev/null | head -1)
    if [ -z "$LOCAL_BACKUP" ]; then
        echo "ERROR: No backup directory found. Specify one: bash scripts/lambda_restore.sh <ip> <backup-dir>"
        exit 1
    fi
fi

echo "=== Restoring to Lambda Instance ==="
echo "  Instance: $IP"
echo "  Backup: $LOCAL_BACKUP"
echo ""

# Upload checkpoints
if [ -d "$LOCAL_BACKUP/checkpoints" ]; then
    echo "Uploading checkpoints..."
    rsync -avz --progress "$LOCAL_BACKUP/checkpoints/" "$REMOTE:$PROJECT/checkpoints/" 2>&1 | tail -3
    echo ""
fi

# Upload logs
if [ -d "$LOCAL_BACKUP/logs" ]; then
    echo "Uploading logs..."
    rsync -avz --progress "$LOCAL_BACKUP/logs/" "$REMOTE:$PROJECT/logs/" 2>&1 | tail -3
    echo ""
fi

# Upload results
if [ -d "$LOCAL_BACKUP/results" ]; then
    echo "Uploading results..."
    rsync -avz --progress "$LOCAL_BACKUP/results/" "$REMOTE:$PROJECT/results/" 2>&1 | tail -3
    echo ""
fi

# Upload data
if [ -d "$LOCAL_BACKUP/data" ]; then
    echo "Uploading data..."
    rsync -avz --progress "$LOCAL_BACKUP/data/" "$REMOTE:$PROJECT/data/" 2>&1 | tail -3
    echo ""
fi

echo "============================================"
echo "Restore complete!"
echo "SSH in and run: bash scripts/lambda_resume.sh"
echo "============================================"
