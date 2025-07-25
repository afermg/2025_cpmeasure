{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        mpkgs = import inputs.nixpkgs_master {
          system = system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        libList = [
          pkgs.stdenv.cc.cc
          pkgs.libGL
          pkgs.glib
          pkgs.libz
        ];
      in
      with pkgs;
      {
        devShells = {
          default =
            let
              # These packages get built by Nix, and will be ahead on the PATH
              pwp = (
                python311.withPackages (
                  p: with p; [
                    python-lsp-server
                    python-lsp-ruff
                    # cython
                  ]
                )
              );
            in
            mkShell {

              packages = [
                pwp
                python311Packages.venvShellHook
                uv
              ] ++ libList;
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
              shellHook = ''
                export UV_PYTHON=${pkgs.python311}
                export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:"/run/opengl-driver/lib":$LD_LIBRARY_PATH
                export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
                export PYTHONPATH=${pwp}/${pwp.sitePackages}:$PYTHONPATH

                runHook venvShellHook
                uv sync

              '';
            };
        };
      }
    );
}
# NIX_LD = runCommand "ld.so" { } ''
#   ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
# '';
