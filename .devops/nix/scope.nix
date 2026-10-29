{
  lib,
  newScope,
  pkgs,
  releaseVersion ? null,
  revision,
  rocmPackages,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit releaseVersion revision rocmPackages; };
  tp2-rdma = self.callPackage ./package.nix {
    inherit releaseVersion revision rocmPackages;
    enableTp2Rdma = true;
    rdma-core = pkgs.rdma-core;
  };
  mkServe = self.callPackage ./mk-serve.nix { };
})
