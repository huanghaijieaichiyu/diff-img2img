#!/bin/bash

VERSION="v1.0.0"
RELEASE_TITLE="Diff-Img2Img Studio v1.0"
DIST_DIR="dist/DiffImg2ImgStudio"
ARCHIVE_NAME="DiffImg2ImgStudio-Linux.tar.gz"

echo "🚀 Preparing Release $VERSION..."

# Check if dist directory exists
if [ ! -d "$DIST_DIR" ]; then
    echo "❌ Build directory not found. Please run ./build_executable.sh first."
    exit 1
fi

# Create Archive
echo "📦 Archiving build..."
tar -czf $ARCHIVE_NAME -C dist DiffImg2ImgStudio

# Check for GitHub CLI
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found."
    echo "👉 Please install it: https://cli.github.com/"
    echo "   Then authenticate with: gh auth login"
    echo "   Or manually upload '$ARCHIVE_NAME' to GitHub Releases."
    exit 1
fi

# Create Release
echo "⬆️  Creating GitHub Release..."
if gh release create "$VERSION" "$ARCHIVE_NAME" --title "$RELEASE_TITLE" --notes "Automated release of Diff-Img2Img Studio."; then
    echo "✅ Release published successfully!"
    echo "🔗 View it here: $(gh repo view --json url -q .url)/releases/tag/$VERSION"
else
    echo "❌ Failed to create release. Ensure you are authenticated and in a git repo."
fi
