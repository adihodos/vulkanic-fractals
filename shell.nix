let
  pkgs = import <nixpkgs> {};
in
pkgs.mkShell {
  buildInputs = [
    pkgs.pkg-config
    pkgs.zlib
    pkgs.vulkan-tools
    pkgs.vulkan-headers
    pkgs.vulkan-loader
    pkgs.vulkan-tools-lunarg
    pkgs.vulkan-validation-layers
    pkgs.shaderc
    pkgs.glslang
    pkgs.xorg.libX11
    pkgs.bashInteractive
  ];

  APPEND_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.vulkan-loader
    pkgs.xorg.libXcursor
    pkgs.xorg.libXi
    pkgs.xorg.libXrandr
  ];
  shellHook = ''
      export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$APPEND_LIBRARY_PATH"
    '';

  # Set Environment Variables
  RUST_BACKTRACE = 1;
}
  
