param(
    [string]$TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'

$taskRoot = (Resolve-Path -LiteralPath $TaskRoot).Path
$outputRoot = Join-Path $taskRoot 'validation\outputs'
$pythonExe = $PythonExe
Set-Location -LiteralPath $taskRoot

function Initialize-StepOutput {
    param(
        [Parameter(Mandatory = $true)][string]$OutputDirectory,
        [Parameter(Mandatory = $true)][string[]]$SourceFiles
    )
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $snapshot = Join-Path $OutputDirectory 'source_snapshot'
    New-Item -ItemType Directory -Path $snapshot -Force | Out-Null
    foreach ($source in $SourceFiles) {
        $sourcePath = Join-Path $taskRoot $source
        $destinationPath = Join-Path $snapshot (Split-Path $source -Leaf)
        if (Test-Path -LiteralPath $destinationPath) {
            $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash
            $destinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destinationPath).Hash
            if ($sourceHash -ne $destinationHash) {
                throw "Source changed after $OutputDirectory was initialized: $source"
            }
        }
        else {
            Copy-Item -LiteralPath $sourcePath -Destination $destinationPath
        }
    }
}

function Invoke-SimulationStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$OutputDirectory,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string[]]$SourceFiles
    )
    Initialize-StepOutput -OutputDirectory $OutputDirectory -SourceFiles $SourceFiles
    $stdout = Join-Path $OutputDirectory 'stdout.log'
    $stderr = Join-Path $OutputDirectory 'stderr.log'
    Write-Output "START $Name $(Get-Date -Format o)"
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $pythonExe @Arguments 1> $stdout 2> $stderr
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorPreference
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $stderr"
    }
    Write-Output "COMPLETE $Name $(Get-Date -Format o)"
}

$commonSources = @(
    'BALL.py',
    'validation\ball_validation_harness.py',
    'validation\direct_rdoc_benchmark.py',
    'validation\direct_rdoc_common.py',
    'validation\direct_rdoc_fair_comparator.py',
    'validation\irt_calibration.json'
)
$seeds = @('1729', '2027', '2028', '2029', '4242', '9001', '13', '71', '137', '311')
$cells = @('linear', 'interaction', 'nonlinear', 'heterogeneous', 'missingness')

$gaussianOut = Join-Path $outputRoot 'direct_rdoc_gaussian_publication_20260805_matched_heldout_v3'
$gaussianArgs = @(
    '-u', 'validation\direct_rdoc_benchmark.py',
    '--cells'
) + $cells + @(
    '--shares', '0.10', '0.25',
    '--seeds'
) + $seeds + @(
    '--n', '150', '--t', '84', '--ensemble-size', '5',
    '--teacher-epochs', '300', '--student-epochs', '300', '--ode-epochs', '300',
    '--anchor-warmup', '120', '--kl-warmup', '80', '--batch-size', '32',
    '--s0-basis', 'matched', '--device', 'cuda', '--out', $gaussianOut, '--resume'
)
Invoke-SimulationStep -Name 'Gaussian benchmark' -OutputDirectory $gaussianOut -Arguments $gaussianArgs -SourceFiles $commonSources

$parameterOut = Join-Path $outputRoot 'focused_parameter_sensitivities_publication_20260805_v1'
$parameterArgs = @(
    '-u', 'validation\run_focused_parameter_sensitivities.py',
    '--out', $parameterOut,
    '--seeds'
) + $seeds + @(
    '--n', '150', '--t', '84', '--ensemble-size', '5',
    '--teacher-epochs', '300', '--student-epochs', '300', '--device', 'cuda'
)
Invoke-SimulationStep -Name 'Focused parameter sensitivities' -OutputDirectory $parameterOut -Arguments $parameterArgs -SourceFiles @(
    'BALL.py',
    'validation\direct_rdoc_benchmark.py',
    'validation\direct_rdoc_fair_comparator.py',
    'validation\run_focused_parameter_sensitivities.py'
)

$calibrationOut = Join-Path $taskRoot 'simulations\paper\outputs\publication_calibration_20260805_v3'
$calibrationArgs = @(
    '-u', 'BALL.py', 'paper-fig3',
    '--replicates', '10', '--n', '1000', '--t', '84', '--base-seed', '60000',
    '--members', '5', '--teacher-epochs', '300', '--student-epochs', '300',
    '--d-model', '96', '--n-layers', '3', '--batch-size', '32', '--device', 'cuda',
    '--out', $calibrationOut
)
Invoke-SimulationStep -Name 'Uncertainty calibration' -OutputDirectory $calibrationOut -Arguments $calibrationArgs -SourceFiles @('BALL.py')

$irtOut = Join-Path $outputRoot 'direct_rdoc_irt_publication_20260805_matched_heldout_v3'
$irtArgs = @(
    '-u', 'validation\direct_rdoc_benchmark.py',
    '--cells'
) + $cells + @(
    '--shares', '0.10', '0.25',
    '--seeds'
) + $seeds + @(
    '--n', '150', '--t', '84', '--ensemble-size', '5',
    '--teacher-epochs', '300', '--student-epochs', '300', '--ode-epochs', '300',
    '--anchor-warmup', '120', '--kl-warmup', '80', '--batch-size', '32',
    '--s0-basis', 'matched', '--device', 'cuda', '--anchor-observation', 'irt',
    '--out', $irtOut, '--resume'
)
Invoke-SimulationStep -Name 'Item-response benchmark' -OutputDirectory $irtOut -Arguments $irtArgs -SourceFiles $commonSources

$adaptiveOut = Join-Path $outputRoot 'direct_rdoc_adaptive_publication_20260805_matched_heldout_v3'
$adaptiveArgs = @(
    '-u', 'validation\direct_rdoc_benchmark.py',
    '--cells'
) + $cells + @(
    '--shares', '0.10', '0.25',
    '--seeds'
) + $seeds + @(
    '--n', '150', '--t', '84', '--ensemble-size', '5',
    '--teacher-epochs', '300', '--student-epochs', '300',
    '--anchor-warmup', '120', '--kl-warmup', '80', '--batch-size', '32',
    '--s0-basis', 'matched', '--skip-gp', '--skip-ode-rnn', '--device', 'cuda',
    '--rdoc-drift-l1', '0.1', '--rdoc-drift-adaptive',
    '--rdoc-drift-adaptive-gamma', '1.0', '--rdoc-drift-adaptive-eps', '0.001',
    '--out', $adaptiveOut, '--resume'
)
Invoke-SimulationStep -Name 'Adaptive coefficient sensitivity' -OutputDirectory $adaptiveOut -Arguments $adaptiveArgs -SourceFiles $commonSources

$controlsOut = Join-Path $outputRoot 'direct_rdoc_negative_controls_publication_20260805_matched_heldout_v3'
$controlSources = $commonSources + @('validation\direct_rdoc_negative_controls.py')
$controlsArgs = @(
    '-u', 'validation\direct_rdoc_negative_controls.py',
    '--seeds'
) + $seeds + @(
    '--n', '150', '--t', '84', '--ensemble-size', '5',
    '--teacher-epochs', '300', '--student-epochs', '300',
    '--anchor-warmup', '120', '--kl-warmup', '80', '--batch-size', '32',
    '--s0-basis', 'matched', '--device', 'cuda', '--out', $controlsOut, '--resume'
)
Invoke-SimulationStep -Name 'Negative controls' -OutputDirectory $controlsOut -Arguments $controlsArgs -SourceFiles $controlSources

Write-Output "SIMULATION CHAIN COMPLETE $(Get-Date -Format o)"
