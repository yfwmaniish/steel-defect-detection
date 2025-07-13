# Steel Defect Detection - Windows Edge Deployment Script

# Set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Set variables
$ConfigDir = "deployment\config"
$ConfigFile = "$ConfigDir\edge_config.yaml"
$ModelDir = "models"
$ModelFile = "$ModelDir\steel_defect_advanced.pt"
$DbDir = "data"
$DbFile = "$DbDir\inspections.db"
$LogDir = "logs"
$LogFile = "$LogDir\edge_detector.log"

# Create necessary directories
New-Item -ItemType Directory -Path $ConfigDir -Force
New-Item -ItemType Directory -Path $ModelDir -Force
New-Item -ItemType Directory -Path $DbDir -Force
New-Item -ItemType Directory -Path $LogDir -Force

# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed. Please install Python 3.8+ from python.org"
    exit 1
}

# Check Python version
$pythonVersion = python --version
Write-Host "Using Python version: $pythonVersion"

# Install Python dependencies
Write-Host "Installing Python dependencies..."
pip install -r deployment\requirements.txt

# Check for model file
if (-not (Test-Path $ModelFile)) {
    Write-Host "ERROR: Model file $ModelFile does not exist. Please provide the trained model file."
    exit 1
}

# Setup SQLite database
if (-not (Test-Path $DbFile)) {
    Write-Host "Initializing database..."
    # Install SQLite if not present
    if (-not (Get-Command sqlite3 -ErrorAction SilentlyContinue)) {
        Write-Host "Installing SQLite..."
        # Download and install SQLite for Windows
        $sqliteUrl = "https://sqlite.org/2023/sqlite-tools-win32-x86-3410000.zip"
        $sqliteZip = "$env:TEMP\sqlite-tools.zip"
        $sqliteDir = "$env:TEMP\sqlite-tools"
        
        Invoke-WebRequest -Uri $sqliteUrl -OutFile $sqliteZip
        Expand-Archive -Path $sqliteZip -DestinationPath $sqliteDir -Force
        
        # Copy sqlite3.exe to system directory or add to PATH
        $sqliteExe = Get-ChildItem -Path $sqliteDir -Recurse -Name "sqlite3.exe"
        Copy-Item -Path "$sqliteDir\$sqliteExe" -Destination "$env:SystemRoot\System32\" -Force
    }
    
    # Initialize database
    Get-Content "deployment\scripts\setup_database.sql" | sqlite3 $DbFile
}

# Create log file
if (-not (Test-Path $LogFile)) {
    New-Item -ItemType File -Path $LogFile -Force
}

# Create Windows service configuration
$serviceName = "SteelDefectDetection"
$serviceDisplayName = "Steel Defect Detection Edge Device"
$serviceDescription = "AI-powered steel defect detection system for quality control"
$pythonPath = (Get-Command python).Source
$scriptPath = Resolve-Path "deployment\edge_detector.py"
$workingDirectory = (Get-Location).Path

# Create service wrapper script
$serviceScript = @"
import sys
import os
import subprocess
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('$LogFile'),
        logging.StreamHandler()
    ]
)

def main():
    try:
        logging.info("Starting Steel Defect Detection Edge Device...")
        os.chdir('$workingDirectory')
        
        # Start the main application
        process = subprocess.Popen([
            '$pythonPath', 
            'deployment/edge_detector.py'
        ], cwd='$workingDirectory')
        
        # Wait for process to complete
        process.wait()
        
    except Exception as e:
        logging.error(f"Error starting service: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"@

$serviceScript | Out-File -FilePath "deployment\service_wrapper.py" -Encoding utf8

# Install as Windows service using NSSM (Non-Sucking Service Manager)
$nssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
$nssmZip = "$env:TEMP\nssm.zip"
$nssmDir = "$env:TEMP\nssm"

Write-Host "Downloading NSSM for Windows service installation..."
Invoke-WebRequest -Uri $nssmUrl -OutFile $nssmZip
Expand-Archive -Path $nssmZip -DestinationPath $nssmDir -Force

$nssmExe = Get-ChildItem -Path $nssmDir -Recurse -Name "nssm.exe" | Where-Object { $_ -match "win64" } | Select-Object -First 1
$nssmPath = "$nssmDir\nssm-2.24\win64\nssm.exe"

if (Test-Path $nssmPath) {
    Write-Host "Installing Windows service..."
    
    # Install service
    & $nssmPath install $serviceName $pythonPath "deployment\service_wrapper.py"
    & $nssmPath set $serviceName DisplayName $serviceDisplayName
    & $nssmPath set $serviceName Description $serviceDescription
    & $nssmPath set $serviceName AppDirectory $workingDirectory
    & $nssmPath set $serviceName Start SERVICE_AUTO_START
    
    # Set service to restart on failure
    & $nssmPath set $serviceName AppExit Default Restart
    & $nssmPath set $serviceName AppRestartDelay 10000
    
    Write-Host "Service installed successfully!"
    Write-Host "To start the service, run: Start-Service $serviceName"
    Write-Host "To stop the service, run: Stop-Service $serviceName"
    
} else {
    Write-Host "WARNING: Could not install Windows service. NSSM not found."
    Write-Host "You can run the edge detector manually with: python deployment\edge_detector.py"
}

# Create startup script
$startupScript = @"
@echo off
cd /d "$workingDirectory"
python deployment\edge_detector.py
pause
"@

$startupScript | Out-File -FilePath "start_edge_detector.bat" -Encoding ascii

# Create configuration validation script
$validateScript = @"
import yaml
import sys
import os

def validate_config():
    config_file = "$ConfigFile"
    
    if not os.path.exists(config_file):
        print(f"ERROR: Config file {config_file} not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['camera', 'ai_model', 'database', 'logging']
        for section in required_sections:
            if section not in config:
                print(f"ERROR: Missing required section: {section}")
                return False
        
        # Validate model file exists
        model_path = config['ai_model']['model_path']
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            return False
        
        print("Configuration validation successful!")
        return True
        
    except Exception as e:
        print(f"ERROR: Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    if validate_config():
        sys.exit(0)
    else:
        sys.exit(1)
"@

$validateScript | Out-File -FilePath "deployment\validate_config.py" -Encoding utf8

# Run configuration validation
Write-Host "Validating configuration..."
python deployment\validate_config.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment completed successfully!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Start the service: Start-Service $serviceName"
    Write-Host "2. Or run manually: python deployment\edge_detector.py"
    Write-Host "3. Monitor logs: Get-Content $LogFile -Wait"
    Write-Host "4. Access web interface: http://localhost:8080"
    Write-Host ""
    Write-Host "Configuration file: $ConfigFile"
    Write-Host "Database file: $DbFile"
    Write-Host "Log file: $LogFile"
} else {
    Write-Host "ERROR: Deployment failed. Please check the configuration and try again."
    exit 1
}
