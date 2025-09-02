{ pkgs, lib, config, inputs, ... }:
let
  buildInputs = with pkgs; [
    cudaPackages.cuda_cudart
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    stdenv.cc.cc
    libuv
    zlib
    # OpenCV stuff
    stdenv.cc.cc.libgcc
    libGL
    mesa
    glib
  ];
in
{
    # https://devenv.sh/basics/
    env = {
        LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"; # For tensorflow with GPU support
        CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
    };

    # https://devenv.sh/packages/
    packages = with pkgs; [
        cudaPackages.cuda_nvcc
        ruff
    ];

    # https://devenv.sh/languages/
    languages.python = {
    enable = true;
    uv = {
        enable = true;
        sync.enable = true;
    };
    };

    # https://devenv.sh/processes/
    processes = {};
    # processes.cargo-watch.exec = "cargo-watch";

    # https://devenv.sh/services/
    # services.postgres.enable = true;

    # https://devenv.sh/scripts/
    enterShell = ''
    . .devenv/state/venv/bin/activate
    nvcc -V
    check-cuda-is-available
    '';

    # https://devenv.sh/tasks/
    # tasks = {
    #   "myproj:setup".exec = "mytool build";
    #   "devenv:enterShell".after = [ "myproj:setup" ];
    # };

    # https://devenv.sh/tests/
#    enterTest = ''
#    echo "Running tests"
#    git --version | grep --color=auto "${pkgs.git.version}"
#    '';

    # https://devenv.sh/git-hooks/
    # git-hooks.hooks.shellcheck.enable = true;

    # See full reference at https://devenv.sh/reference/options/
}
