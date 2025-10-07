{
  description = "YOLOv11 Face Detection development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = false;
          };
        };
        
        ultralytics-thop = ps: ps.buildPythonPackage rec {
          pname = "ultralytics-thop";
          version = "2.0.11";
          format = "pyproject";
          
          src = pkgs.fetchFromGitHub {
            owner = "ultralytics";
            repo = "thop";
            rev = "v${version}";
            hash = "sha256-f3Lg/sgYlMvu2/7EDXda43ZLVOnXmWkOc29rmQYR34g=";
          };
          
          nativeBuildInputs = with ps; [ setuptools ];
          
          propagatedBuildInputs = with ps; [
            pytorch
          ];
          
          doCheck = false;
          
          meta = with pkgs.lib; {
            description = "FLOPs counter for PyTorch models";
            homepage = "https://github.com/ultralytics/thop";
            license = licenses.mit;
          };
        };
        
        ultralytics = ps: ps.buildPythonPackage rec {
          pname = "ultralytics";
          version = "8.3.61";
          format = "pyproject";
          
          src = ps.fetchPypi {
            inherit pname version;
            hash = "sha256-bL7RXyRH/5PTfK+mHxgWylvzM77ItzojeeqOh83FzK8=";
          };
          
          nativeBuildInputs = with ps; [ setuptools ];
          
          propagatedBuildInputs = with ps; [
            numpy
            opencv4
            pillow
            pytorch
            torchvision
            pyyaml
            tqdm
            requests
            huggingface-hub
            matplotlib
            pandas
            seaborn
            py-cpuinfo
            (ultralytics-thop ps)
          ];
          
          pythonImportsCheck = [ "ultralytics" ];
          doCheck = false;
          dontCheckRuntimeDeps = true;
          
          meta = with pkgs.lib; {
            description = "Ultralytics YOLO";
            homepage = "https://github.com/ultralytics/ultralytics";
            license = licenses.agpl3Only;
          };
        };
        
        finalPython = pkgs.python3.withPackages (ps: [
          (ultralytics ps)
          (ultralytics-thop ps)
        ] ++ (with ps; [
          numpy
          opencv4
          pillow
          pytorch
          torchvision
          pyyaml
          tqdm
          requests
          huggingface-hub
          matplotlib
          pandas
          seaborn
          py-cpuinfo
        ]));
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            finalPython
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.ffmpeg
            pkgs.mpv
          ];
          
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.stdenv.cc.cc.lib
            ]}:$LD_LIBRARY_PATH
            
            echo "CUDA environment loaded"
            echo "Run: python detect_faces_yolo11.py --source <video_file> --output <output_file>"
            echo "     Add --play to watch with mpv after processing"
          '';
        };
      }
    );
}

