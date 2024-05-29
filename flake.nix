{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python312.withPackages (python-pkgs: [
            python-pkgs.numpy
            python-pkgs.matplotlib
            python-pkgs.graphviz
          ]))

          graphviz
        ];

        shellHook = ''
          tmux -L autograd
          exit
        '';
      };
    };
}
