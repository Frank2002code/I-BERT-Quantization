#!/bin/bash

COMMIT_MSG="${1:-$(date +"%Y-%m-%d %H:%M:%S")}"
git pull
git status
git add -A
git commit -m "$COMMIT_MSG" || echo "No changes to commit"
git push -u main ibert