param(
    [string]$TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [Parameter(Mandatory = $true)][string]$ProtectedRoot,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'

$taskRoot = (Resolve-Path -LiteralPath $TaskRoot).Path
$protectedRoot = (Resolve-Path -LiteralPath $ProtectedRoot).Path
$protectedRun = Join-Path $protectedRoot 'publication_ready_20260805_v3'
$stabilityOut = Join-Path $protectedRoot 'publication_stability_20260805_v3'
$rdocScorerOut = Join-Path $protectedRoot 'rdoc_scorer_sensitivity_20260805_v3'
$comorbidityFeatures = Join-Path $protectedRoot 'session_comorbidity_features.csv'
$publicationOut = Join-Path $taskRoot 'simulations\paper\outputs\publication_ready_20260805_v3'
$calibration = Join-Path $taskRoot 'simulations\paper\outputs\publication_calibration_20260805_v3'
$pythonExe = $PythonExe

$primary = Join-Path $taskRoot 'validation\outputs\direct_rdoc_gaussian_publication_20260805_matched_heldout_v3'
$irt = Join-Path $taskRoot 'validation\outputs\direct_rdoc_irt_publication_20260805_matched_heldout_v3'
$adaptive = Join-Path $taskRoot 'validation\outputs\direct_rdoc_adaptive_publication_20260805_matched_heldout_v3'
$controls = Join-Path $taskRoot 'validation\outputs\direct_rdoc_negative_controls_publication_20260805_matched_heldout_v3'
$parameterSensitivity = Join-Path $taskRoot 'validation\outputs\focused_parameter_sensitivities_publication_20260805_v1'

function Assert-SimulationOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][int]$ExpectedRows,
        [Parameter(Mandatory = $true)][int]$ExpectedMethods
    )
    foreach ($name in @('per_run.csv', 'aggregate.csv', 'manifest.json')) {
        $path = Join-Path $Directory $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Simulation output is incomplete: $path"
        }
    }
    $rows = Import-Csv -LiteralPath (Join-Path $Directory 'per_run.csv')
    if ($rows.Count -ne $ExpectedRows) {
        throw "Expected $ExpectedRows simulation rows in $Directory; found $($rows.Count)"
    }
    $methods = @($rows | Select-Object -ExpandProperty method -Unique)
    if ($methods.Count -ne $ExpectedMethods) {
        throw "Expected $ExpectedMethods simulation methods in $Directory; found $($methods.Count)"
    }
    $counts = $rows | Group-Object method
    $expectedPerMethod = [int]($ExpectedRows / $ExpectedMethods)
    if (@($counts | Where-Object { $_.Count -ne $expectedPerMethod }).Count -gt 0) {
        throw "Simulation methods do not have identical replicate counts in $Directory"
    }
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Stdout,
        [Parameter(Mandatory = $true)][string]$Stderr
    )
    Write-Output "START $Name $(Get-Date -Format o)"
    $priorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $pythonExe @Arguments 1> $Stdout 2> $Stderr
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $priorPreference
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $Stderr"
    }
    Write-Output "COMPLETE $Name $(Get-Date -Format o)"
}

Set-Location -LiteralPath $taskRoot
Assert-SimulationOutput -Directory $primary -ExpectedRows 1100 -ExpectedMethods 11
Assert-SimulationOutput -Directory $irt -ExpectedRows 1100 -ExpectedMethods 11
Assert-SimulationOutput -Directory $adaptive -ExpectedRows 900 -ExpectedMethods 9
Assert-SimulationOutput -Directory $controls -ExpectedRows 200 -ExpectedMethods 5
foreach ($name in @('persistence_per_run.csv', 'persistence_development_selected.csv', 'anchor_weight_per_run.csv', 'manifest.json')) {
    $path = Join-Path $parameterSensitivity $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Focused parameter sensitivity output is incomplete: $path"
    }
}

if (-not (Test-Path -LiteralPath $comorbidityFeatures -PathType Leaf)) {
    throw "Protected comorbidity feature file is absent: $comorbidityFeatures"
}
foreach ($path in @($protectedRun, $stabilityOut, $rdocScorerOut, $publicationOut)) {
    if (Test-Path -LiteralPath $path) {
        throw "Release target already exists; choose a new versioned target: $path"
    }
}
New-Item -ItemType Directory -Path $protectedRun | Out-Null

$empiricalArgs = @(
    '-u', 'BALL.py', 'empirical-fit',
    '--run-label', 'publication_ready_20260805_v3',
    '--run-dir', $protectedRun,
    '--cadences', '14', '21', '28',
    '--teacher-epochs', '120',
    '--student-epochs', '120',
    '--anchor-warmup', '60',
    '--kl-warmup', '60',
    '--members', '5',
    '--d-model', '96',
    '--n-heads', '4',
    '--n-layers', '3',
    '--batch-size', '32',
    '--ode-hidden', '96',
    '--ode-epochs', '120',
    '--ode-members', '5',
    '--selection-epochs', '120',
    '--selection-members', '1',
    '--device', 'cuda',
    '--comorbidity-features', $comorbidityFeatures,
    '--direct-causal-ablation',
    '--dense-ehr-ablation',
    '--anchor-only-ablation',
    '--gp-benchmark',
    '--ode-rnn-benchmark',
    '--generalization-sensitivity',
    '--rdoc-transition-analysis',
    '--select-core-hyperparameters',
    '--distillation-decomposition',
    '--compute-matched-direct',
    '--session-balanced-anchors',
    '--no-publish-top-level'
)
Invoke-PythonStep -Name 'Empirical benchmark and ablation suite' -Arguments $empiricalArgs -Stdout (Join-Path $protectedRun 'empirical_stdout.log') -Stderr (Join-Path $protectedRun 'empirical_stderr.log')

$stabilityArgs = @(
    '-u', 'validation\analyze_empirical_stability.py',
    '--transitions', (Join-Path $protectedRun 'empirical_ball_transition_rows_complete.csv'),
    '--out', $stabilityOut
)
New-Item -ItemType Directory -Path $stabilityOut | Out-Null
Invoke-PythonStep -Name 'Within-patient stability analysis' -Arguments $stabilityArgs -Stdout (Join-Path $protectedRun 'stability_stdout.log') -Stderr (Join-Path $protectedRun 'stability_stderr.log')

$rdocArgs = @(
    '-u', 'validation\analyze_rdoc_scorer_sensitivity.py',
    '--transitions', (Join-Path $protectedRun 'empirical_rdoc_transition_rows_complete.csv'),
    '--ensemble-summary', (Join-Path $protectedRun 'empirical_rdoc_transition_complete.csv'),
    '--ensemble-coefficients', (Join-Path $protectedRun 'empirical_rdoc_transition_coefficients_complete.csv'),
    '--out', $rdocScorerOut
)
New-Item -ItemType Directory -Path $rdocScorerOut | Out-Null
Invoke-PythonStep -Name 'RDoC scorer sensitivity' -Arguments $rdocArgs -Stdout (Join-Path $protectedRun 'rdoc_scorer_stdout.log') -Stderr (Join-Path $protectedRun 'rdoc_scorer_stderr.log')

$validationArgs = @(
    '-u', 'validation\validate_publication_outputs.py',
    '--benchmark', $primary,
    '--calibration', $calibration,
    '--empirical', $protectedRun,
    '--controls', $controls,
    '--adaptive', $adaptive,
    '--irt', $irt
)
Invoke-PythonStep -Name 'Analysis-result validation' -Arguments $validationArgs -Stdout (Join-Path $protectedRun 'analysis_validation.json') -Stderr (Join-Path $protectedRun 'analysis_validation_stderr.log')

$buildArgs = @(
    '-u', 'validation\build_direct_rdoc_manuscript_outputs.py',
    '--primary', $primary,
    '--controls', $controls,
    '--adaptive', $adaptive,
    '--irt', $irt,
    '--calibration', $calibration,
    '--empirical', $protectedRun,
    '--stability', $stabilityOut,
    '--rdoc-scorer', $rdocScorerOut,
    '--out', $publicationOut
)
Invoke-PythonStep -Name 'Aggregate manuscript table and figure build' -Arguments $buildArgs -Stdout (Join-Path $protectedRun 'publication_build_stdout.log') -Stderr (Join-Path $protectedRun 'publication_build_stderr.log')

$packageValidationArgs = @(
    '-u', 'validation\validate_manuscript_output_package.py',
    '--package', $publicationOut
)
Invoke-PythonStep -Name 'Aggregate publication-package validation' -Arguments $packageValidationArgs -Stdout (Join-Path $protectedRun 'publication_package_validation_stdout.log') -Stderr (Join-Path $protectedRun 'publication_package_validation_stderr.log')

Write-Output "EMPIRICAL AND PUBLICATION CHAIN COMPLETE $(Get-Date -Format o)"
Write-Output "Protected empirical run: $protectedRun"
Write-Output "Aggregate manuscript package: $publicationOut"
