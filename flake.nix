{
  description = "A Nix-flake-based Rust development environment";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f {
          pkgs = import nixpkgs {
            inherit system;
            overlays = [rust-overlay.overlays.default self.overlays.default];
          };
        });
  in {
    overlays.default = final: prev: {
      rustToolchain = let
        rust = prev.rust-bin;
      in
        if builtins.pathExists ./rust-toolchain.toml
        then rust.fromRustupToolchainFile ./rust-toolchain.toml
        else if builtins.pathExists ./rust-toolchain
        then rust.fromRustupToolchainFile ./rust-toolchain
        else
          rust.stable.latest.default.override {
            extensions = ["rust-src" "rustfmt"];
          };
    };

    devShells = forEachSupportedSystem ({pkgs}: {
      default = pkgs.mkShell {
        packages = with pkgs; [
          rustToolchain
          openssl
          pkg-config
          cargo-deny
          cargo-edit
          cargo-watch
          rust-analyzer
          gdb
          cmake
          cmakeCurses
          gf
          helix

          zlib
          vulkan-tools
          vulkan-headers
          vulkan-loader
          vulkan-tools-lunarg
          vulkan-validation-layers
          vulkan-caps-viewer
          vulkan-utility-libraries
          shaderc
          glslang
          xorg.libX11
          python311
          renderdoc

          shaderc
          shaderc.bin
          shaderc.static
          shaderc.dev
          shaderc.lib
        ];

        APPEND_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.xorg.libXcursor
          pkgs.xorg.libXi
          pkgs.xorg.libX11
          pkgs.xorg.libXrandr
          pkgs.xorg.libXext
          pkgs.xorg.libXxf86vm
          pkgs.libxkbcommon
          pkgs.xorg.libxcb.dev
          pkgs.shaderc.lib
          pkgs.shaderc.dev
          pkgs.shaderc.static
          pkgs.glslang
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$APPEND_LIBRARY_PATH"
        '';

        env = {
          # Required by rust-analyzer
          RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
          RUST_BACKTRACE = 1;
        };
      };
    });
  };
}
