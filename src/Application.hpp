#pragma once
#ifndef APPLICATION_HPP_
#define APPLICATION_HPP_

#include <SDL2/SDL.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <optional>
#include <vulkan/vulkan.hpp>

const int MAX_FRAMES_IN_FLIGHT = 2;

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  [[nodiscard]] auto IsComplete() const -> bool {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct SwapchainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilites;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

struct Vertex {
  glm::vec3 Position = glm::vec3(0.0F);
  glm::vec4 Color = glm::vec4(1.0F);

  static auto GetBindingDescription() -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription bindingDescription(0, sizeof(Vertex),
                                                         vk::VertexInputRate::eVertex);
    return bindingDescription;
  }

  static auto GetAttributeDescriptions() -> std::array<vk::VertexInputAttributeDescription, 2> {
    std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};

    attributeDescriptions[0] = vk::VertexInputAttributeDescription(
        0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, Position));
    attributeDescriptions[1] = vk::VertexInputAttributeDescription(
        1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, Color));

    return attributeDescriptions;
  }
};

const std::vector<Vertex> vertices = {{{-0.5F, -0.5F, 0.0F}, {1.0F, 0.0F, 0.0F, 1.0F}},
                                      {{0.5F, -0.5F, 0.0F}, {0.0F, 1.0F, 0.0F, 1.0F}},
                                      {{0.5F, 0.5F, 0.0F}, {0.0F, 0.0F, 1.0F, 1.0F}},
                                      {{-0.5F, 0.5F, 0.0F}, {1.0F, 1.0F, 1.0F, 1.0F}}};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

class Application {
  const std::vector<const char *> requiredDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  const uint32_t WIDTH = 1600;
  const uint32_t HEIGHT = 900;

 public:
  void Run();

 private:
  static auto ReadFile(const std::string &filename) -> std::vector<uint8_t>;

  void InitWindow();
  void InitVulkan();
  void MainLoop();

  void UpdateUniformBuffers(uint32_t currentImage);

  void Cleanup();

  void CreateInstance();
  // void SetupDebugMessenger();
  void RecreateSwapchain();
  void CleanupSwapchain();
  void CreateSwapchain();
  void CreateImageViews();
  void CreateRenderPass();
  void CreateDescriptorSetLayout();
  void CreateGraphicsPipeline();
  void CreateFramebuffers();
  void CreateUniformBuffers();
  void CreateDescriptorPool();
  void CreateDescriptorSets();
  void CreateCommandBuffers();

  auto IsDeviceSuitable(const vk::PhysicalDevice &device) -> bool;
  auto FindQueueFamilies(const vk::PhysicalDevice &device) -> QueueFamilyIndices;
  auto CheckExtensionSupport(const vk::PhysicalDevice &device) -> bool;
  auto QuerySwapchainSupportDetails(const vk::PhysicalDevice &device) -> SwapchainSupportDetails;
  auto ChooseSwapchainFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
      -> vk::SurfaceFormatKHR;
  auto ChooseSwapchainPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
      -> vk::PresentModeKHR;
  auto ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilites) -> vk::Extent2D;

  auto FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) -> uint32_t;

  auto CreateShaderModule(const std::vector<uint8_t> &code) -> vk::ShaderModule;

  auto CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties) -> std::pair<vk::Buffer, vk::DeviceMemory>;
  void CopyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size);

  // Vulkan
  vk::Instance instance;
  vk::PhysicalDevice physicalDevice;
  vk::Device device;
  vk::SwapchainKHR swapchain;
  vk::SurfaceKHR surface = VK_NULL_HANDLE;
  vk::Queue presentQueue = VK_NULL_HANDLE;
  vk::Queue graphicsQueue = VK_NULL_HANDLE;
  vk::Format swapchainFormat;
  vk::Extent2D swapchainExtent;
  std::vector<vk::Image> swapchainImages;
  std::vector<vk::ImageView> swapchainImageViews;

  vk::Pipeline graphicsPipeline;
  vk::RenderPass renderPass;
  vk::DescriptorPool descriptorPool;
  std::vector<vk::DescriptorSet> descriptorSets;
  vk::DescriptorSetLayout descriptorSetLayout;
  vk::PipelineLayout pipelineLayout;

  std::vector<vk::Framebuffer> swapchainFramebuffers;

  vk::CommandPool commandPool;
  std::vector<vk::CommandBuffer> commandBuffers;

  vk::Buffer vertexBuffer;
  vk::DeviceMemory vertexBufferMemory;
  vk::Buffer indexBuffer;
  vk::DeviceMemory indexBufferMemory;

  std::vector<vk::Buffer> uniformBuffers;
  std::vector<vk::DeviceMemory> uniformBuffersMemory;

  size_t currentFrame = 0;
  std::array<vk::Fence, MAX_FRAMES_IN_FLIGHT> inFlightFences;
  std::vector<vk::Fence> imagesInFlight;
  std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> imageAvailableSempaphores;
  std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> renderFinishedSemaphores;

  bool framebufferResized = false;

  // SDL
  SDL_Window *window = nullptr;

  bool running = true;
};

#endif  // APPLICATION_HPP_
