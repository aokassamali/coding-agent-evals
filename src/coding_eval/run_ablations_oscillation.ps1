# Config paths
$MODEL_DIR = "C:\Users\aokas\localmodel"
$TASKS_PATH = "data/tasks/v11_osc_total_200.jsonl"
$MAX_ATTEMPTS = 4
$TIMEOUT_S = 3
$VARIANTS = "B_debug"
$PORT = 8080
# Reduced to 4K to ensure 7B fits safely on 1080 Ti
$CONTEXT = 4096  

$models = Get-ChildItem -Path $MODEL_DIR -Filter "*.gguf"

foreach ($model in $models) {
    $rawName = $model.BaseName
    $modelId = $rawName -replace '[-\.]', '_'
    $runTag = "zz_$modelId"
    
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "Testing: $rawName" -ForegroundColor Green
    Write-Host "========================================"
    
    # 1. Start llama-server with merged stdout/stderr logging
    $logDir = "runs\_server_logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logPath = Join-Path $logDir "$runTag.log"

    $serverProc = Start-Process -FilePath "cmd" `
        -ArgumentList "/c llama-server -m `"$($model.FullName)`" -c $CONTEXT --n-gpu-layers 99 --port $PORT > `"$logPath`" 2>&1" `
        -PassThru

    # 2. Health check loop (wait for actual ready state)
    Write-Host "Waiting for server to be ready..." -NoNewline
    $maxTries = 30  # 30*2sec = 60sec max wait
    $ready = $false
    for ($i = 0; $i -lt $maxTries; $i++) {
        Start-Sleep -Seconds 2
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/health" -TimeoutSec 1 -ErrorAction Stop
            if ($resp.status -eq "ok" -or $resp.status -eq "healthy") {
                $ready = $true
                Write-Host " READY!" -ForegroundColor Green
                break
            }
        } catch {
            Write-Host "." -NoNewline
        }
        
        # Check if process crashed
        if ($serverProc.HasExited) {
            Write-Host "`nServer crashed during load! Check $logPath" -ForegroundColor Red
            break
        }
    }
    
    if (-not $ready) {
        if (-not $serverProc.HasExited) { $serverProc | Stop-Process -Force }
        Write-Host "Skipping $modelId (server failed to start)" -ForegroundColor Yellow
        continue
    }
    
    # 3. Test a simple completion first (smoke test)
    try {
        $testBody = @{
            messages = @(@{role="user"; content="hi"})
            max_tokens = 10
        } | ConvertTo-Json -Compress
        
        $testResp = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/v1/chat/completions" `
            -Method POST -Body $testBody -ContentType "application/json" -TimeoutSec 10
        
        Write-Host "Smoke test passed" -ForegroundColor Cyan
    } catch {
        Write-Host "Smoke test failed: $_. Skipping." -ForegroundColor Red
        $serverProc | Stop-Process -Force
        continue
    }
    
    # 4. Fingerprint server via chat completion
    $nonce = [Guid]::NewGuid().ToString("N").Substring(0,8)

    $fingerprintPrompt = @"
You are a fingerprinting endpoint.
Return EXACTLY this text and nothing else:
FINGERPRINT:$nonce
"@

    $body = @{
        messages = @(
            @{ role = "system"; content = "Return exactly what the user asks. No extra text." },
            @{ role = "user"; content  = $fingerprintPrompt }
        )
        max_tokens = 32
        temperature = 0
    } | ConvertTo-Json -Compress

    try {
        $fpResp = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/v1/chat/completions" `
            -Method POST -Body $body -ContentType "application/json" -TimeoutSec 15

        $fpText = $fpResp.choices[0].message.content.Trim()
        Write-Host "Fingerprint response: $fpText" -ForegroundColor Cyan
    } catch {
        Write-Host "Fingerprint chat call failed: $_" -ForegroundColor Red
        $fpText = "ERROR"
    }

    Write-Host "Model file: $($model.FullName)" -ForegroundColor Gray
    Write-Host "Fingerprint nonce: $nonce" -ForegroundColor Gray

    # 5. Run actual eval
    Write-Host "Running eval for $modelId..." -ForegroundColor Yellow
    try {
        $evalArgs = @(
            ".\src\coding_eval\run_eval.py",
            "--tasks_path", $TASKS_PATH,
            "--model_id", $modelId,
            "--run_tag", $runTag,
            "--max_attempts", $MAX_ATTEMPTS,
            "--timeout-s", $TIMEOUT_S,
            "--variants", $VARIANTS
        )
        python @evalArgs
        Write-Host "Eval complete" -ForegroundColor Green
    } catch {
        Write-Host "Eval error: $_" -ForegroundColor Red
    }
    
    # 6. Cleanup with proper verification
    Write-Host "Shutting down..." -NoNewline
    $serverProc | Stop-Process -Force -ErrorAction SilentlyContinue
    $serverProc.WaitForExit()  # Ensures process handle is closed

    # Wait for port to be released
    $maxWait = 15
    for ($i = 0; $i -lt $maxWait; $i++) {
        $portInUse = Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue
        if (-not $portInUse) { 
            Write-Host " Port released" -ForegroundColor Gray
            break 
        }
        Start-Sleep -Seconds 1
    }

    if ($i -eq $maxWait) {
        Write-Host " Warning: Port may still be in use" -ForegroundColor Yellow
    }

    Write-Host ""
}

Write-Host "All done!" -ForegroundColor Green
