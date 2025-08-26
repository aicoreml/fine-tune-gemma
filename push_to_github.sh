#!/bin/bash
# push_to_github.sh - Script to push the repository to GitHub

echo "Pushing repository to GitHub..."

# Set your GitHub username and repository name here
USERNAME="your-github-username"
REPO_NAME="gemma-finetuning"

echo "Adding remote origin..."
git remote add origin https://github.com/$USERNAME/$REPO_NAME.git

echo "Pushing to GitHub..."
git push -u origin main

echo "Done! Repository pushed to GitHub."