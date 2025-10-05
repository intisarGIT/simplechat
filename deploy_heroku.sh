#!/usr/bin/env bash
# Usage: ./deploy_heroku.sh <heroku-app-name>
# This script assumes you have the Heroku CLI installed and you're logged in.
if [ -z "$1" ]; then
  echo "Usage: $0 <heroku-app-name>"
  exit 1
fi
APP=$1

# Create app if not exists
heroku apps:info -a "$APP" || heroku create "$APP"

# Push to heroku
git push heroku main

# Set config vars from local .env (do NOT commit .env to git). This reads and sets each non-empty var.
if [ -f ".env" ]; then
  echo "Setting config vars from .env"
  # Remove comments and empty lines, then export
  export $(grep -v '^#' .env | xargs)
  heroku config:set MISTRAL_API_KEY="$MISTRAL_API_KEY" IMAGEROUTER_API_KEY="$IMAGEROUTER_API_KEY" FACEPP_API_KEY="$FACEPP_API_KEY" FACEPP_API_SECRET="$FACEPP_API_SECRET" -a "$APP"
else
  echo ".env not found — remember to set required config vars via heroku config:set"
fi

echo "Done. Tail logs with: heroku logs --tail -a $APP"
