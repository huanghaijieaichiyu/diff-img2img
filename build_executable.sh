#!/bin/bash
echo "🚀 Starting Build Process..."

# Install PyInstaller if not present
if ! command -v pyinstaller &> /dev/null; then
    echo "📦 Installing PyInstaller..."
    pip install pyinstaller
fi

echo "🏗️  Building Executable..."
pyinstaller build.spec --clean --noconfirm

echo "✅ Build Complete!"
echo "📂 Executable is located in: dist/DiffImg2ImgStudio/DiffImg2ImgStudio"
echo "💡 To run: ./dist/DiffImg2ImgStudio/DiffImg2ImgStudio"
