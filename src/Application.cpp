#include "Application.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "Assert.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

void Application::Run() {
  InitWindow();
  InitVulkan();
  MainLoop();
  Cleanup();
}

void Application::InitWindow() {
  SDL_Init(SDL_INIT_VIDEO);

  window = SDL_CreateWindow("dvulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                            static_cast<int32_t>(WIDTH), static_cast<int32_t>(HEIGHT),
                            SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN);
}

void Application::InitVulkan() {
  CreateInstance();

  // TODO Setup Debugging

  {
    // Surface creation
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window, instance, &surface))
      throw std::runtime_error("Failed to create surface");
    this->surface = surface;
  }

  {
    // Picking physical device
    auto devices = instance.enumeratePhysicalDevices();

    for (const auto &d : devices) {
      if (IsDeviceSuitable(d)) {
        physicalDevice = d;
        break;
      }
    }
  }

  {
    // Create Logical Device
    auto indices = FindQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};

    float queuePrio = 1.f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1, &queuePrio);
      queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::DeviceCreateInfo createInfo(vk::DeviceCreateFlags(), queueCreateInfos, {},
                                    requiredDeviceExtensions, {});
    device = physicalDevice.createDevice(createInfo);

    graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
    presentQueue = device.getQueue(indices.presentFamily.value(), 0);
  }

  CreateSwapchain();
  CreateImageViews();
  CreateRenderPass();
  CreateDescriptorSetLayout();
  CreateGraphicsPipeline();
  CreateFramebuffers();

  {
    // Create Command Pool
    QueueFamilyIndices queueIndices = FindQueueFamilies(physicalDevice);
    vk::CommandPoolCreateInfo poolInfo(vk::CommandPoolCreateFlags(),
                                       queueIndices.graphicsFamily.value());
    commandPool = device.createCommandPool(poolInfo);
  }

  {
    // Create Vertex Buffer
    vk::DeviceSize bufSize = sizeof(Vertex) * vertices.size();

    auto [stagingBuf, stagingMem] = CreateBuffer(
        bufSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    uint8_t *data = static_cast<uint8_t *>(device.mapMemory(stagingMem, 0, bufSize));
    memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
    device.unmapMemory(stagingMem);

    std::tie(vertexBuffer, vertexBufferMemory) = CreateBuffer(
        bufSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    CopyBuffer(stagingBuf, vertexBuffer, bufSize);
  }

  {
    // Index buffer
    vk::DeviceSize bufSize = sizeof(indices[0]) * indices.size();

    auto [stagingBuf, stagingMem] = CreateBuffer(
        bufSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = device.mapMemory(stagingMem, 0, bufSize);
    memcpy(data, indices.data(), bufSize);
    device.unmapMemory(stagingMem);

    std::tie(indexBuffer, indexBufferMemory) = CreateBuffer(
        bufSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    CopyBuffer(stagingBuf, indexBuffer, bufSize);
  }

  CreateUniformBuffers();
  CreateDescriptorPool();
  CreateDescriptorSets();

  CreateCommandBuffers();

  {
    imagesInFlight.resize(swapchainImages.size(), VK_NULL_HANDLE);

    // Create Sync Objects
    vk::SemaphoreCreateInfo semaphoreInfo((vk::SemaphoreCreateFlags()));
    vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      imageAvailableSempaphores.at(i) = device.createSemaphore(semaphoreInfo);
      renderFinishedSemaphores.at(i) = device.createSemaphore(semaphoreInfo);
      inFlightFences.at(i) = device.createFence(fenceInfo);
    }
  }
}

void Application::MainLoop() {
  while (running) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      switch (e.type) {
        case SDL_QUIT:
          running = false;
          break;
        case SDL_WINDOWEVENT_RESIZED:
          framebufferResized = true;
          break;
      }
    }

    while (vk::Result::eTimeout ==
           device.waitForFences(inFlightFences.at(currentFrame), VK_TRUE, UINT64_MAX))
      ;

    auto currentBuffer = device.acquireNextImageKHR(swapchain, UINT64_MAX,
                                                    imageAvailableSempaphores.at(currentFrame));

    if (currentBuffer.result == vk::Result::eErrorOutOfDateKHR) {
      RecreateSwapchain();
      continue;
    } else if (currentBuffer.result != vk::Result::eSuccess &&
               currentBuffer.result != vk::Result::eSuboptimalKHR) {
      throw std::runtime_error("Failed to acquire swapchain image");
    }

    uint32_t imageIndex = currentBuffer.value;

    UpdateUniformBuffers(imageIndex);

    if (imagesInFlight.at(imageIndex))
      while (vk::Result::eTimeout ==
             device.waitForFences(imagesInFlight.at(imageIndex), true, UINT64_MAX))
        ;

    imagesInFlight[imageIndex] = inFlightFences.at(currentFrame);

    std::array<vk::Semaphore, 1> signalSemaphores = {renderFinishedSemaphores.at(currentFrame)};
    std::array<vk::PipelineStageFlags, 1> waitStages = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput};

    vk::SubmitInfo submitInfo(imageAvailableSempaphores.at(currentFrame), waitStages,
                              commandBuffers, signalSemaphores);

    device.resetFences(inFlightFences.at(currentFrame));

    graphicsQueue.submit(submitInfo, inFlightFences.at(currentFrame));

    // Present
    vk::PresentInfoKHR presentInfo(signalSemaphores, swapchain, imageIndex);
    vk::Result result = presentQueue.presentKHR(presentInfo);

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR ||
        framebufferResized) {
      framebufferResized = false;
      RecreateSwapchain();
    } else if (result != vk::Result::eSuccess)
      throw std::runtime_error("Failed to present swapchain image");

    presentQueue.waitIdle();
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  device.waitIdle();
}

void Application::UpdateUniformBuffers(uint32_t currentImage) {
  static auto startTime = std::chrono::high_resolution_clock::now();

  auto currentTime = std::chrono::high_resolution_clock::now();
  float time =
      std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

  UniformBufferObject ubo{};
  ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.proj =
      glm::perspective(glm::radians(45.0f),
                       (float)swapchainExtent.width / (float)swapchainExtent.height, 0.1f, 1000.0f);
  ubo.proj[1][1] *= -1;

  void *data = device.mapMemory(uniformBuffersMemory[currentImage], 0, sizeof(ubo));
  memcpy(data, &ubo, sizeof(ubo));
  device.unmapMemory(uniformBuffersMemory[currentImage]);
}

void Application::Cleanup() { SDL_Quit(); }

void Application::CreateInstance() {
  // Extensions
  uint32_t extensionCount = 0;
  SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr);
  std::vector<const char *> extensions(extensionCount);
  SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensions.data());

  // Layers
  const std::array<const char *, 1> layers = {"VK_LAYER_KHRONOS_validation"};

  // Create Instance
  vk::ApplicationInfo appInfo("dvulkan", 1, "dvulkan", 1, VK_API_VERSION_1_2);
  vk::InstanceCreateInfo createInfo({}, &appInfo, layers, extensions);
  instance = vk::createInstance(createInfo);
}

void Application::RecreateSwapchain() {
  vkDeviceWaitIdle(device);

  CleanupSwapchain();

  CreateSwapchain();
  CreateImageViews();
  CreateRenderPass();
  CreateGraphicsPipeline();
  CreateFramebuffers();
  CreateUniformBuffers();
  CreateDescriptorPool();
  CreateCommandBuffers();
}

void Application::CleanupSwapchain() {
  for (size_t i = 0; i < swapchainFramebuffers.size(); i++)
    device.destroyFramebuffer(swapchainFramebuffers[i]);

  device.freeCommandBuffers(commandPool, commandBuffers);

  device.destroyPipeline(graphicsPipeline);
  device.destroyPipelineLayout(pipelineLayout);
  device.destroyRenderPass(renderPass);
  for (size_t i = 0; i < swapchainImageViews.size(); i++)
    device.destroyImageView(swapchainImageViews[i]);

  device.destroySwapchainKHR(swapchain);
}

void Application::CreateSwapchain() {
  // Create swapchain
  SwapchainSupportDetails details = QuerySwapchainSupportDetails(physicalDevice);
  auto format = ChooseSwapchainFormat(details.formats);
  auto presentMode = ChooseSwapchainPresentMode(details.presentModes);
  auto extent = ChooseSwapExtent(details.capabilites);

  swapchainFormat = format.format;
  swapchainExtent = extent;

  uint32_t imageCount = details.capabilites.minImageCount + 1;
  if (details.capabilites.maxImageCount > 0 && imageCount > details.capabilites.maxImageCount)
    imageCount = details.capabilites.maxImageCount;

  QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);
  std::vector<uint32_t> queueFamilyIndices = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};

  vk::SharingMode sharingMode = vk::SharingMode::eConcurrent;
  if (indices.graphicsFamily == indices.presentFamily) {
    sharingMode = vk::SharingMode::eExclusive;
    queueFamilyIndices.clear();
  }

  vk::SwapchainCreateInfoKHR createInfo({}, surface, imageCount, format.format, format.colorSpace,
                                        extent, 1, vk::ImageUsageFlagBits::eColorAttachment,
                                        sharingMode, queueFamilyIndices,
                                        details.capabilites.currentTransform,
                                        vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, true);

  swapchain = device.createSwapchainKHR(createInfo);
  swapchainImages = device.getSwapchainImagesKHR(swapchain);
}

void Application::CreateImageViews() {
  // Swapchain Image Views
  swapchainImageViews.resize(swapchainImages.size());
  for (size_t i = 0; i < swapchainImages.size(); i++) {
    vk::ImageViewCreateInfo createInfo(
        {}, swapchainImages[i], vk::ImageViewType::e2D, swapchainFormat,
        {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
         vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    swapchainImageViews[i] = device.createImageView(createInfo);
  }
}

void Application::CreateRenderPass() {
  vk::AttachmentDescription colorAttachment(
      vk::AttachmentDescriptionFlags(), swapchainFormat, vk::SampleCountFlagBits::e1,
      vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
      vk::ImageLayout::ePresentSrcKHR);

  vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);
  vk::SubpassDescription subpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics,
                                 {}, colorAttachmentRef);

  // vk::SubpassDependency dependency(0, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput,
  //                                  vk::PipelineStageFlagBits::eColorAttachmentOutput,
  //                                  vk::AccessFlagBits::eNoneKHR,
  //                                  vk::AccessFlagBits::eColorAttachmentWrite);

  vk::RenderPassCreateInfo createInfo({}, colorAttachment, subpass);

  renderPass = device.createRenderPass(createInfo);
}

void Application::CreateDescriptorSetLayout() {
  vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                                  vk::ShaderStageFlagBits::eVertex);

  vk::DescriptorSetLayoutCreateInfo layoutInfo(vk::DescriptorSetLayoutCreateFlags(),
                                               uboLayoutBinding);
  descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
}

void Application::CreateDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(swapchainImages.size(), descriptorSetLayout);
  vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, layouts);
  descriptorSets = device.allocateDescriptorSets(allocInfo);

  for (size_t i = 0; i < swapchainImages.size(); i++) {
    vk::DescriptorBufferInfo bufInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));

    vk::WriteDescriptorSet descriptorWrite(
        descriptorSets[i], 0, 0, vk::DescriptorType::eUniformBuffer, nullptr, bufInfo, nullptr);

    device.updateDescriptorSets(descriptorWrite, {});
  }
}

void Application::CreateGraphicsPipeline() {
  auto vertShaderCode = ReadFile("assets/vert.spv");
  auto fragShaderCode = ReadFile("assets/frag.spv");
  auto vertShaderModule = CreateShaderModule(vertShaderCode);
  auto fragShaderModule = CreateShaderModule(fragShaderCode);

  vk::PipelineShaderStageCreateInfo vertCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                   vk::ShaderStageFlagBits::eVertex,
                                                   vertShaderModule, "main");

  vk::PipelineShaderStageCreateInfo fragCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                   vk::ShaderStageFlagBits::eFragment,
                                                   fragShaderModule, "main");

  std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {vertCreateInfo, fragCreateInfo};

  auto bindingDescription = Vertex::GetBindingDescription();
  auto attributeDescriptions = Vertex::GetAttributeDescriptions();

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo(vk::PipelineVertexInputStateCreateFlags(),
                                                         bindingDescription, attributeDescriptions);

  vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
      vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList, false);

  vk::Viewport viewport(0, 0, static_cast<float>(swapchainExtent.width),
                        static_cast<float>(swapchainExtent.height), 0.0f, 1.0f);
  vk::Rect2D scissor({0, 0}, swapchainExtent);

  vk::PipelineViewportStateCreateInfo viewportState(vk::PipelineViewportStateCreateFlags(),
                                                    viewport, scissor);

  vk::PipelineRasterizationStateCreateInfo rasterizer(
      vk::PipelineRasterizationStateCreateFlags(), false, false, vk::PolygonMode::eFill,
      vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

  vk::PipelineMultisampleStateCreateInfo multisampling(vk::PipelineMultisampleStateCreateFlags(),
                                                       vk::SampleCountFlagBits::e1, false, 1.0f,
                                                       nullptr, false, false);

  vk::PipelineColorBlendAttachmentState colorBlendAttachment(
      false, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

  vk::PipelineColorBlendStateCreateInfo colorBlending(
      vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eNoOp, colorBlendAttachment);

  std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamicState(vk::PipelineDynamicStateCreateFlags(),
                                                  dynamicStates);

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo(vk::PipelineLayoutCreateFlags(),
                                                  descriptorSetLayout);
  pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

  vk::GraphicsPipelineCreateInfo pipelineInfo(vk::PipelineCreateFlags(), shaderStages,
                                              &vertexInputInfo, &inputAssembly, nullptr,
                                              &viewportState, &rasterizer, &multisampling, nullptr,
                                              &colorBlending, nullptr, pipelineLayout, renderPass);

  vk::Result result{};
  std::tie(result, graphicsPipeline) = device.createGraphicsPipeline(nullptr, pipelineInfo);
}

void Application::CreateFramebuffers() {
  // Create Framebuffers
  swapchainFramebuffers.resize(swapchainImageViews.size());
  for (size_t i = 0; i < swapchainImageViews.size(); i++) {
    std::array<vk::ImageView, 1> attachments = {swapchainImageViews[i]};
    vk::FramebufferCreateInfo fbInfo(vk::FramebufferCreateFlags(), renderPass, attachments,
                                     swapchainExtent.width, swapchainExtent.height, 1);
    swapchainFramebuffers[i] = device.createFramebuffer(fbInfo);
  }
}

void Application::CreateUniformBuffers() {
  vk::DeviceSize bufSize = sizeof(UniformBufferObject);

  uniformBuffers.resize(swapchainImages.size());
  uniformBuffersMemory.resize(swapchainImages.size());

  for (size_t i = 0; i < swapchainImages.size(); i++) {
    std::tie(uniformBuffers[i], uniformBuffersMemory[i]) = CreateBuffer(
        bufSize, vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  }
}

void Application::CreateDescriptorPool() {
  vk::DescriptorPoolSize poolSize(vk::DescriptorType::eUniformBuffer,
                                  static_cast<uint32_t>(swapchainImages.size()));

  vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlags(),
                                        static_cast<uint32_t>(swapchainImages.size()), poolSize);
  descriptorPool = device.createDescriptorPool(poolInfo);
}

void Application::CreateCommandBuffers() {
  // Create Command Buffers
  vk::CommandBufferAllocateInfo allocInfo(commandPool, vk::CommandBufferLevel::ePrimary,
                                          swapchainFramebuffers.size());
  commandBuffers = device.allocateCommandBuffers(allocInfo);

  // Begin Recording command buffers
  for (size_t i = 0; i < commandBuffers.size(); i++) {
    vk::CommandBufferBeginInfo beginInfo;

    vk::ClearValue clearValue(vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.2f, 1.0f}));
    vk::RenderPassBeginInfo renderPassInfo(renderPass, swapchainFramebuffers[i],
                                           vk::Rect2D({0, 0}, swapchainExtent), clearValue);

    commandBuffers[i].begin(beginInfo);
    commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    commandBuffers[i].bindVertexBuffers(0, vertexBuffer, {0});
    commandBuffers[i].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);

    commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                         descriptorSets[i], {});
    commandBuffers[i].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    commandBuffers[i].endRenderPass();
    commandBuffers[i].end();
  }
}

bool Application::IsDeviceSuitable(const vk::PhysicalDevice &device) {
  auto indices = FindQueueFamilies(device);

  bool extensionsSupported = CheckExtensionSupport(device);

  bool swapchainAdequate = false;
  if (extensionsSupported) {
    auto details = QuerySwapchainSupportDetails(device);
    swapchainAdequate = !details.formats.empty() && !details.presentModes.empty();
  }

  return indices.IsComplete() && extensionsSupported && swapchainAdequate;
}

QueueFamilyIndices Application::FindQueueFamilies(const vk::PhysicalDevice &device) {
  QueueFamilyIndices indices{};

  auto queueFamilies = device.getQueueFamilyProperties();

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) indices.graphicsFamily = i;
    auto presentSupport = device.getSurfaceSupportKHR(i, surface);

    if (presentSupport) indices.presentFamily = i;
    if (indices.IsComplete()) break;

    i++;
  }

  return indices;
}

bool Application::CheckExtensionSupport(const vk::PhysicalDevice &device) {
  auto availableExtensions = device.enumerateDeviceExtensionProperties();

  std::set<std::string> deviceExtensions(requiredDeviceExtensions.begin(),
                                         requiredDeviceExtensions.end());

  for (const auto &e : availableExtensions) deviceExtensions.erase(e.extensionName);
  return deviceExtensions.empty();
}

SwapchainSupportDetails Application::QuerySwapchainSupportDetails(
    const vk::PhysicalDevice &device) {
  SwapchainSupportDetails details;

  details.capabilites = device.getSurfaceCapabilitiesKHR(surface);

  details.formats = device.getSurfaceFormatsKHR(surface);
  details.presentModes = device.getSurfacePresentModesKHR(surface);

  return details;
}

vk::SurfaceFormatKHR Application::ChooseSwapchainFormat(
    const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
  for (const auto &format : availableFormats) {
    if (format.format == vk::Format::eB8G8R8Srgb &&
        format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
      return format;
  }

  return availableFormats[0];
}

vk::PresentModeKHR Application::ChooseSwapchainPresentMode(
    const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  for (const auto &mode : availablePresentModes) {
    if (mode == vk::PresentModeKHR::eMailbox) return mode;
  }

  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Application::ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilites) {
  if (capabilites.currentExtent.width != UINT32_MAX) return capabilites.currentExtent;

  int width = 0, height = 0;
  SDL_GetWindowSize(window, &width, &height);

  vk::Extent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

  actualExtent.width = std::clamp(actualExtent.width, capabilites.minImageExtent.width,
                                  capabilites.maxImageExtent.width);
  actualExtent.height = std::clamp(actualExtent.height, capabilites.minImageExtent.height,
                                   capabilites.maxImageExtent.height);

  return actualExtent;
}

uint32_t Application::FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
  auto memProperties = physicalDevice.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties))
      return i;
  }

  throw std::runtime_error("Failed to find suitable memory type");
}

vk::ShaderModule Application::CreateShaderModule(const std::vector<uint8_t> &code) {
  vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(), code.size(),
                                        reinterpret_cast<const uint32_t *>(code.data()));
  return device.createShaderModule(createInfo);
}

std::pair<vk::Buffer, vk::DeviceMemory> Application::CreateBuffer(
    vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {
  vk::BufferCreateInfo bufferInfo(vk::BufferCreateFlags(), size, usage,
                                  vk::SharingMode::eExclusive);
  auto buf = device.createBuffer(bufferInfo);

  vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(buf);
  vk::MemoryAllocateInfo allocInfo(memReqs.size,
                                   FindMemoryType(memReqs.memoryTypeBits, properties));
  auto mem = device.allocateMemory(allocInfo);

  device.bindBufferMemory(buf, mem, 0);

  return {buf, mem};
}

void Application::CopyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size) {
  vk::CommandBufferAllocateInfo allocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);

  std::vector<vk::CommandBuffer> buf = device.allocateCommandBuffers(allocInfo);

  vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  buf[0].begin(beginInfo);

  vk::BufferCopy copyRegion(0, 0, size);
  buf[0].copyBuffer(src, dst, copyRegion);

  buf[0].end();

  vk::SubmitInfo submitInfo(0, {}, {}, 1, buf.data(), 0, {});
  graphicsQueue.submit(submitInfo);
  graphicsQueue.waitIdle();
}

std::vector<uint8_t> Application::ReadFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::in);
  if (!file.is_open()) throw std::runtime_error("failed to open file");

  file.seekg(0, std::ios::end);
  size_t filesize = file.tellg();

  std::vector<char> buf(filesize);
  file.seekg(0, std::ios::beg);
  file.read(buf.data(), filesize);

  file.close();

  return std::vector<uint8_t>(buf.begin(), buf.end());
}
