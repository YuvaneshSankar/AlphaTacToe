#!/bin/bash
# Clean up script for AlphaTacToe repository

echo "🧹 Cleaning up AlphaTacToe repository..."

# Remove Python cache files
echo "Removing __pycache__ directories..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove compiled Python files
echo "Removing .pyc files..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove training checkpoints (optional - comment out if you want to keep them)
# echo "Removing training checkpoints..."
# rm -rf checkpoints/ 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.temp" -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "📁 Repository is clean and ready for git!"
echo "💡 Tip: Run 'git status' to see what will be committed"



#Before committing to git:
# ./cleanup.sh
# git add .
# git commit -m """