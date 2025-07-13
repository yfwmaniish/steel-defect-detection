# PowerShell script to convert XML annotations to YOLO format

# Class mapping
$classMapping = @{
    'crazing' = 0        # scratch
    'inclusion' = 5      # patch
    'patches' = 5        # patch
    'pitted_surface' = 4 # hole
    'rolled-in_scale' = 3 # color_defect
    'scratches' = 0      # scratch
}

function Convert-XMLToYOLO {
    param(
        [string]$xmlPath,
        [string]$outputPath
    )
    
    try {
        [xml]$xml = Get-Content $xmlPath
        $annotations = @()
        
        $width = [int]$xml.annotation.size.width
        $height = [int]$xml.annotation.size.height
        
        foreach ($object in $xml.annotation.object) {
            $className = $object.name
            $classId = $classMapping[$className]
            
            if ($null -ne $classId) {
                $xmin = [int]$object.bndbox.xmin
                $xmax = [int]$object.bndbox.xmax
                $ymin = [int]$object.bndbox.ymin
                $ymax = [int]$object.bndbox.ymax
                
                # Convert to YOLO format (normalized)
                $xCenter = ($xmin + $xmax) / 2.0 / $width
                $yCenter = ($ymin + $ymax) / 2.0 / $height
                $bboxWidth = ($xmax - $xmin) / $width
                $bboxHeight = ($ymax - $ymin) / $height
                
                $annotations += "$classId $xCenter $yCenter $bboxWidth $bboxHeight"
            }
        }
        
        if ($annotations.Count -gt 0) {
            $annotations | Out-File -FilePath $outputPath -Encoding ASCII
        }
    }
    catch {
        Write-Host "Error processing $xmlPath : $_"
    }
}

# Process training annotations
Write-Host "Converting training annotations..."
$trainAnnotations = Get-ChildItem "data\downloaded\NEU-DET\train\annotations\*.xml"
foreach ($xmlFile in $trainAnnotations) {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($xmlFile.Name)
    $outputFile = "data\labeled\train\labels\$baseName.txt"
    Convert-XMLToYOLO -xmlPath $xmlFile.FullName -outputPath $outputFile
}

# Process validation annotations
Write-Host "Converting validation annotations..."
$valAnnotations = Get-ChildItem "data\downloaded\NEU-DET\validation\annotations\*.xml"
foreach ($xmlFile in $valAnnotations) {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($xmlFile.Name)
    $outputFile = "data\labeled\validation\labels\$baseName.txt"
    Convert-XMLToYOLO -xmlPath $xmlFile.FullName -outputPath $outputFile
}

Write-Host "Conversion completed!"
Write-Host "Training labels: $(Get-ChildItem 'data\labeled\train\labels\*.txt' | Measure-Object).Count"
Write-Host "Validation labels: $(Get-ChildItem 'data\labeled\validation\labels\*.txt' | Measure-Object).Count"
