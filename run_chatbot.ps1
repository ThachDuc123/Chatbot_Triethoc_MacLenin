param(
    [int]$BackendPort = 5002,
    [int]$UiPort = 3000,
    [string]$OllamaModel = "qwen2.5:7b-instruct",
    [string]$EmbeddingBackend = "fastembed",
    [string]$EmbeddingModel = "BAAI/bge-small-en-v1.5",
    [int]$TopK = 4,
    # Set 0 or -1 for "freestyle" (server will use a large safe default).
    [int]$MaxNewTokens = 768,
    [int]$OllamaTimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"

function Test-PortInUse([int]$Port) {
    try {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        return ($null -ne $c)
    } catch {
        return $false
    }
}

function Start-Backend {
    if (Test-PortInUse $BackendPort) {
        # Port is in use, but it might not be our Flask app. Verify health.
        $ok = Wait-Http "http://localhost:$BackendPort/" 3
        if ($ok) {
            Write-Host "[backend] Backend already responding on http://localhost:$BackendPort/." -ForegroundColor Yellow
            return
        }
        Write-Host "[backend] Port $BackendPort is in use but backend is not responding. Trying to start/restart..." -ForegroundColor Yellow
    }

    Write-Host "[backend] Starting Flask backend on http://localhost:$BackendPort ..." -ForegroundColor Cyan

    $args = @(
        "serve.py",
        "--host", "127.0.0.1",
        "--port", "$BackendPort",
        "--mode", "offline",
        "--model_engine", "ollama",
        "--model_name", $OllamaModel,
        "--embedding_backend", $EmbeddingBackend,
        "--embedding_model", $EmbeddingModel,
        "--top_k", "$TopK"
    )

    if ($MaxNewTokens -gt 0) {
        $args += @("--max_new_tokens", "$MaxNewTokens")
    } else {
        Write-Host "[backend] MaxNewTokens <= 0 => freestyle mode (server uses safe default)." -ForegroundColor Yellow
        $args += @("--max_new_tokens", "0")
    }

    # Ensure long generations have enough time on slower machines.
    $env:OLLAMA_TIMEOUT_SECONDS = "$OllamaTimeoutSeconds"
    Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory $PSScriptRoot -WindowStyle Normal
}

function Start-Ui {
    if (Test-PortInUse $UiPort) {
        Write-Host "[ui] Port $UiPort is already in use. Assuming UI server is running." -ForegroundColor Yellow
        return
    }

    Write-Host "[ui] Starting static UI on http://localhost:$UiPort ..." -ForegroundColor Cyan
    Start-Process -FilePath "python" -ArgumentList @("-m", "http.server", "$UiPort") -WorkingDirectory $PSScriptRoot -WindowStyle Normal
}

function Wait-Http([string]$Url, [int]$Seconds = 30) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri $Url
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

function Wait-PortListen([int]$Port, [int]$Seconds = 30) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
        try {
            $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
            if ($null -ne $c) {
                return $true
            }
        } catch {
            # ignore
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

Start-Backend
Start-Ui

# Give the backend a moment to bind the port.
$portOk = Wait-PortListen $BackendPort 30
if (-not $portOk) {
    Write-Host "[warn] Backend port $BackendPort is not listening yet. It may still be starting or it may have crashed." -ForegroundColor Yellow
}

$backendOk = Wait-Http "http://localhost:$BackendPort/" 60
if (-not $backendOk) {
    Write-Host "[warn] Backend didn't respond on http://localhost:$BackendPort/ yet. It may still be starting." -ForegroundColor Yellow
}

$uiUrl = "http://localhost:$UiPort/index.html"
Write-Host "[open] Opening $uiUrl" -ForegroundColor Green
Start-Process $uiUrl

Write-Host "\nDone. Close the opened terminals to stop servers." -ForegroundColor Green
