param(
    [switch]$Sllm,
    [switch]$Vllm,
    [switch]$Build,
    [switch]$Binary
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir

$composeArgs = @("-f", "docker-compose.yml", "--env-file", ".env", "--env-file", ".env.llamacpp-gguf")

$services = @("mariadb", "api")
if ($Sllm) {
    $composeArgs = @("-f", "docker-compose.yml", "--env-file", ".env", "--env-file", ".env.llamacpp-gguf", "--profile", "sllm")
    $services = @("mariadb", "sllm-llamacpp", "api")
}
if ($Vllm) {
    $composeArgs = @("-f", "docker-compose.yml", "--env-file", ".env", "--env-file", ".env.vllm", "--profile", "sllm-vllm")
    $services = @("mariadb", "sllm-vllm", "api")
}

if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "required file not found: docker-compose.yml"
}
$envFiles = @(".env") + ($composeArgs | Where-Object { $_ -like ".env.*" })
foreach ($envFile in $envFiles) {
    if (-not (Test-Path $envFile)) {
        Write-Error "required file not found: $envFile"
    }
}

if ($Build) {
    if ($Binary) {
        $env:ASR_SUMMARY_DOCKERFILE = "Dockerfile.binary"
        Write-Host "[start] building binary api-gpu image"
    } else {
        Write-Host "[start] building api-gpu image"
    }
    docker compose @composeArgs build api
}

Write-Host "[start] starting $($services -join ' ')"
docker compose @composeArgs up -d @services

Write-Host "[start] status"
docker compose @composeArgs ps
