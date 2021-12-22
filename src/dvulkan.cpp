#include "Application.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
  try {
    auto app = std::make_unique<Application>();
    app->Run();
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
