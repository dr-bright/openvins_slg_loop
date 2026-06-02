/*
 * ov_dino backend smoke test.
 */

#include "track/dino_backend.h"

#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  ov_dino::dino_config config;
  if (argc >= 2) {
    config.model = argv[1];
  }
  if (argc >= 3) {
    config.device = argv[2];
  }
  config.verbosity = "info";

  try {
    ov_dino::dino_backend backend(config, false);
    std::cout << "DINO backend engine is ready.\n";
  } catch (const std::exception &e) {
    std::cerr << "dino_backend_smoke_test failed: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
