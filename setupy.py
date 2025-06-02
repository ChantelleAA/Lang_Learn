#!/usr/bin/env python3
"""
Setup script for Visual & Audio Translator
This script helps install and configure the required dependencies
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def install_python_packages():
    """Install required Python packages"""
    packages = [
        "flask",
        "flask-cors", 
        "pillow",
        "opencv-python",
        "numpy",
        "requests",
        "SpeechRecognition",
        "torch",
        "transformers",
        "werkzeug"
    ]
    
    print("Installing Python packages...")
    for package in packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Optional packages for better audio support
    optional_packages = [
        "PyAudio",  # For microphone input
        "pydub",    # For audio processing
    ]
    
    print("\nInstalling optional packages for enhanced audio support...")
    for package in optional_packages:
        run_command(f"pip install {package}", f"Installing {package} (optional)")

def setup_ffmpeg_windows():
    """Setup FFmpeg on Windows"""
    print("\n=== Setting up FFmpeg for Windows ===")
    
    # Check if FFmpeg is already available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is already installed and available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Download and setup FFmpeg
    ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    download_path = "ffmpeg.zip"
    extract_path = "ffmpeg"
    
    try:
        print("📥 Downloading FFmpeg...")
        urllib.request.urlretrieve(ffmpeg_url, download_path)
        
        print("📦 Extracting FFmpeg...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find the ffmpeg.exe file
        for root, dirs, files in os.walk(extract_path):
            if "ffmpeg.exe" in files:
                ffmpeg_exe_path = os.path.join(root, "ffmpeg.exe")
                ffmpeg_dir = root
                
                # Add to PATH for current session
                current_path = os.environ.get("PATH", "")
                if ffmpeg_dir not in current_path:
                    os.environ["PATH"] = current_path + os.pathsep + ffmpeg_dir
                
                print(f"✅ FFmpeg installed at: {ffmpeg_exe_path}")
                print(f"Added to PATH: {ffmpeg_dir}")
                
                # Cleanup
                os.remove(download_path)
                
                return True
        
        print("❌ Could not find ffmpeg.exe in extracted files")
        return False
        
    except Exception as e:
        print(f"❌ Error setting up FFmpeg: {e}")
        return False

def setup_ffmpeg_linux():
    """Setup FFmpeg on Linux"""
    print("\n=== Setting up FFmpeg for Linux ===")
    
    # Try different package managers
    commands = [
        "sudo apt update && sudo apt install -y ffmpeg",
        "sudo yum install -y ffmpeg",
        "sudo dnf install -y ffmpeg",
        "sudo pacman -S ffmpeg"
    ]
    
    for cmd in commands:
        if run_command(cmd, "Installing FFmpeg"):
            return True
    
    print("❌ Could not install FFmpeg automatically")
    print("Please install FFmpeg manually using your system's package manager")
    return False

def setup_ffmpeg_mac():
    """Setup FFmpeg on macOS"""
    print("\n=== Setting up FFmpeg for macOS ===")
    
    # Try Homebrew first
    if run_command("brew install ffmpeg", "Installing FFmpeg with Homebrew"):
        return True
    
    # Try MacPorts
    if run_command("sudo port install ffmpeg", "Installing FFmpeg with MacPorts"):
        return True
    
    print("❌ Could not install FFmpeg automatically")
    print("Please install Homebrew and run: brew install ffmpeg")
    return False

def setup_ffmpeg():
    """Setup FFmpeg based on the operating system"""
    system = platform.system().lower()
    
    if system == "windows":
        return setup_ffmpeg_windows()
    elif system == "linux":
        return setup_ffmpeg_linux()
    elif system == "darwin":  # macOS
        return setup_ffmpeg_mac()
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "static/crops",
        "uploads",
        "output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_system_requirements():
    """Check system requirements"""
    print("=== Checking System Requirements ===")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor} (supported)")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
        return False
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 4:
            print(f"✅ RAM: {memory_gb:.1f} GB (sufficient)")
        else:
            print(f"⚠️  RAM:")