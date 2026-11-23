[CmdletBinding()]
param(
    [string]$Source,
    [string]$Output,
    [int]$Seed = 42,
    [string[]]$Extensions = @('.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'),
    [switch]$DryRun
)

# Compatible with Windows PowerShell 5.1. No external dependencies.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $Source) { $Source = Join-Path $PSScriptRoot '..\Invoice High Accuracy v2' }
if (-not $Output) { $Output = Join-Path $PSScriptRoot '..\data_split' }
$sourcePath = (Resolve-Path -LiteralPath $Source).ProviderPath.TrimEnd('\', '/')
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output).TrimEnd('\', '/')
if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
    throw 'Source must be a directory.'
}
if ($outputPath.Equals($sourcePath, [StringComparison]::OrdinalIgnoreCase) -or
    $outputPath.StartsWith($sourcePath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must be outside Source to avoid including previous splits.'
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Output already exists: $outputPath. Choose a new output directory; existing data is never overwritten."
}
$allowedExtensions = @($Extensions | ForEach-Object {
    if ($_.StartsWith('.')) { $_.ToLowerInvariant() } else { '.' + $_.ToLowerInvariant() }
})
$files = @(Get-ChildItem -LiteralPath $sourcePath -Recurse -File |
    Where-Object { $_.Extension.ToLowerInvariant() -in $allowedExtensions } |
    Sort-Object FullName)
if ($files.Count -eq 0) { throw 'No matching document files found.' }

# Group byte-identical documents so duplicates cannot cross split boundaries.
$groupsByHash = @{}
foreach ($file in $files) {
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    if (-not $groupsByHash.ContainsKey($hash)) {
        $groupsByHash[$hash] = [Collections.Generic.List[object]]::new()
    }
    $groupsByHash[$hash].Add($file)
}
$random = [Random]::new($Seed)
$groups = @(foreach ($hash in @($groupsByHash.Keys | Sort-Object)) {
    [pscustomobject]@{ Hash = $hash; Files = $groupsByHash[$hash]; RandomOrder = $random.NextDouble() }
})
# Place larger duplicate groups first; seed randomizes groups of equal size.
$groups = @($groups | Sort-Object @{Expression = { $_.Files.Count }; Descending = $true}, RandomOrder, Hash)
$names = @('train', 'valid', 'test')
$ratios = @(0.70, 0.15, 0.15)
$targets = @{}
$counts = @{}
$remainders = @()
for ($i = 0; $i -lt $names.Count; $i++) {
    $raw = $files.Count * $ratios[$i]
    $targets[$names[$i]] = [int][Math]::Floor($raw)
    $counts[$names[$i]] = 0
    $remainders += [pscustomobject]@{ Name = $names[$i]; Fraction = $raw - [Math]::Floor($raw); Index = $i }
}
# Largest remainder rounding gives an integer total equal to the file count.
$remaining = $files.Count - ($targets.Values | Measure-Object -Sum).Sum
$ranked = @($remainders | Sort-Object @{Expression = 'Fraction'; Descending = $true}, Index)
for ($i = 0; $i -lt $remaining; $i++) { $targets[$ranked[$i].Name]++ }

$manifest = [Collections.Generic.List[object]]::new()
foreach ($group in $groups) {
    $size = $group.Files.Count
    $candidates = @($names | Where-Object { ($targets[$_] - $counts[$_]) -ge $size })
    if ($candidates.Count -eq 0) {
        # Very large duplicate groups can make exact file counts impossible.
        $candidates = $names
    }
    $split = $candidates | Sort-Object @{Expression = { $targets[$_] - $counts[$_] }; Descending = $true},
        @{Expression = { [Array]::IndexOf($names, $_) }} | Select-Object -First 1
    $counts[$split] += $size
    foreach ($file in $group.Files) {
        $relative = $file.FullName.Substring($sourcePath.Length + 1)
        $destination = Join-Path $split $relative
        $vendorFolder = if ($relative.Contains('\')) { $relative.Split('\')[0] } else { '' }
        $manifest.Add([pscustomobject]@{
            source_relative_path = $relative
            split = $split
            output_relative_path = $destination
            vendor_folder = $vendorFolder
            sha256 = $group.Hash
            duplicate_group_size = $size
            bytes = $file.Length
            seed = $Seed
        })
    }
}

$splitSummary = @(foreach ($name in $names) {
    [pscustomobject]@{
        split = $name
        target_count = $targets[$name]
        file_count = $counts[$name]
        percentage = [Math]::Round(100.0 * $counts[$name] / $files.Count, 2)
    }
})
$splitSummary | Format-Table -AutoSize | Out-Host
$duplicateGroups = @($groups | Where-Object { $_.Files.Count -gt 1 }).Count
Write-Host "Files: $($files.Count); unique contents: $($groups.Count); duplicate groups: $duplicateGroups; seed: $Seed"
foreach ($name in $names) {
    if ($targets[$name] -ne $counts[$name]) {
        Write-Warning 'Split counts differ from rounded targets because identical files are kept together.'
        break
    }
}
if ($DryRun) {
    Write-Host 'Dry run: no files or directories written.'
    return
}

New-Item -ItemType Directory -Path $outputPath | Out-Null
foreach ($name in $names) {
    New-Item -ItemType Directory -Path (Join-Path $outputPath $name) | Out-Null
}
foreach ($row in $manifest) {
    $destination = Join-Path $outputPath $row.output_relative_path
    $parent = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath (Join-Path $sourcePath $row.source_relative_path) -Destination $destination
}
$manifest | Sort-Object split, source_relative_path |
    Export-Csv -LiteralPath (Join-Path $outputPath 'split_manifest.csv') -NoTypeInformation -Encoding UTF8
$summary = [ordered]@{
    source = $sourcePath
    seed = $Seed
    ratios = [ordered]@{ train = 0.70; valid = 0.15; test = 0.15 }
    total_files = $files.Count
    unique_contents = $groups.Count
    duplicate_groups = $duplicateGroups
    splits = $splitSummary
    strategy = 'Seeded random file split; SHA256-identical files kept in the same split.'
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputPath 'split_summary.json') -Encoding UTF8
Write-Host "Created dataset split: $outputPath"
