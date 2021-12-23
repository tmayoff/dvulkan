#pragma once
#ifndef APPLICATION_HPP_
#define APPLICATION_HPP_

#include <SDL2/SDL.h>

#include <optional>
#include <vulkan/vulkan.hpp>

const int MAX_FRAMES_IN_FLIGHT = 2;

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool IsComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapchainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilites;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

class Application {
  const std::vector<const char *> requiredDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  const uint32_t WIDTH = 1600;
  const uint32_t HEIGHT = 900;

 public:
  void Run();

 private:
  static std::vector<uint8_t> ReadFile(const std::string &filename);

  void InitWindow();
  void InitVulkan();
  void MainLoop();
  void Cleanup();

  // Swapchain
  void RecreateSwapchain();
  void CleanupSwapchain();
  void CreateSwapchain();
  void CreateImageViews();
  void CreateRenderPass();
  void CreateGraphicsPipeline();
  void CreateFramebuffers();
  void CreateCommandBuffers();

  bool IsDeviceSuitable(VkPhysicalDevice device);
  QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);
  bool CheckExtensionSupport(VkPhysicalDevice device);
  SwapchainSupportDetails QuerySwapchainSupportDetails(VkPhysicalDevice device);
  VkSurfaceFormatKHR ChooseSwapchainFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);
  VkPresentModeKHR ChooseSwapchainPresentMode(
      const std::vector<VkPresentModeKHR> &availablePresentModes);
  VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilites);

  VkShaderModule CreateShaderModule(const std::vector<uint8_t> &code);

  // Vulkan
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue presentQueue = VK_NULL_HANDLE;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swapchainFormat;
  VkExtent2D swapchainExtent;
  std::vector<VkImage> swapchainImages;
  std::vector<VkImageView> swapchainImageViews;

  VkPipeline graphicsPipeline;
  VkRenderPass renderPass;
  VkPipelineLayout pipelineLayout;

  std::vector<VkFramebuffer> swapchainFramebuffers;

  VkCommandPool commandPool;
  std::vector<VkCommandBuffer> commandBuffers;

  size_t currentFrame = 0;
  std::array<VkFence, MAX_FRAMES_IN_FLIGHT> inFlightFences;
  std::vector<VkFence> imagesInFlight;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> imageAvailableSempaphores;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> renderFinishedSemaphores;

  bool framebufferResized = false;

  // SDL
  SDL_Window *window = nullptr;

  bool running = true;
};

#endif  // APPLICATION_HPP_
