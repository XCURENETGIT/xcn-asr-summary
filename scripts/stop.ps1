param(
    [switch]$Sllm,
    [switch]$Vllm,
    [switch]$Volumes
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir

$composeArgs = @("-f", "docker-compose.yml", "--env-file", ".env", "--env-file", ".env.llamacpp-gguf")
if ($Sllm) {
    $composeArgs = @("-f", "docker-compose.yml", "--env-file", ".env", "--env-file", ".env.llamacpp-gguf")
}
if ($Vllm) {
    $composeArgs = @("-f", "docker-compose.yml", "--env-file", ".env", "--env-file", ".env.vllm", "--profile", "sllm-vllm")
}

if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "required file not found: docker-compose.yml"
}

if ($Volumes) {
    Write-Host "[stop] stopping and removing containers with volumes"
    docker compose @composeArgs down -v
} else {
    Write-Host "[stop] stopping and removing containers"
    docker compose @composeArgs down
}

Write-Host "[stop] done"
