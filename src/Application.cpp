#include "Application.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <array>
#include <iostream>
#include <set>

#include "Assert.hpp"

void Application::Run() {
  InitWindow();
  InitVulkan();
  MainLoop();
  Cleanup();
}

void Application::InitWindow() {
  SDL_Init(SDL_INIT_VIDEO);

  window = SDL_CreateWindow("dvulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH,
                            HEIGHT, SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN);
}

void Application::InitVulkan() {
  {
    // Extensions
    uint32_t extensionCount;
    SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr);
    std::vector<const char *> extensions(extensionCount);
    SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensions.data());

    // Layers
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    std::cout << "Layers";
    for (auto l : availableLayers) {
      std::cout << "\tLayer: " << l.layerName << std::endl;
    }

    const std::array<const char *, 1> layers = {"VK_LAYER_KHRONOS_validation"};

    // Create Instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "dvulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "dvulkan";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = layers.size();
    createInfo.ppEnabledLayerNames = layers.data();

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
      throw std::runtime_error("Failed to create instance");
  }

  {
      // Setup Debugging
  }

  {
    // Surface creation
    if (!SDL_Vulkan_CreateSurface(window, instance, &surface))
      throw std::runtime_error("Failed to create surface");
  }

  {
    // Picking physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("Failed to find GPU with vulkan support");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto &d : devices) {
      if (IsDeviceSuitable(d)) {
        physicalDevice = d;
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE)
      throw std::runtime_error("Failed to find suitable device");
  }

  {
    // Create Logical Device
    auto indices = FindQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};

    float queuePrio = 1.f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePrio;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = requiredDeviceExtensions.size();
    createInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
      throw std::runtime_error("Failed to create logical device");

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
  }

  {
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

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = format.format;
    createInfo.imageColorSpace = format.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);
    std::array<uint32_t, 2> queueFamilyIndices = {indices.graphicsFamily.value(),
                                                  indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = details.capabilites.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS)
      throw std::runtime_error("Failed to create swapchain");

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
  }

  {
    // Swapchain Image Views
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); i++) {
      VkImageViewCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.format = swapchainFormat;
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      if (vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS)
        throw std::runtime_error("failed to create image views");
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
      }
    }
  }
}

void Application::Cleanup() {
  for (auto imageView : swapchainImageViews) vkDestroyImageView(device, imageView, nullptr);
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);
  SDL_Quit();
}

bool Application::IsDeviceSuitable(VkPhysicalDevice device) {
  auto indices = FindQueueFamilies(device);

  bool extensionsSupported = CheckExtensionSupport(device);

  bool swapchainAdequate = false;
  if (extensionsSupported) {
    auto details = QuerySwapchainSupportDetails(device);
    swapchainAdequate = !details.formats.empty() && !details.presentModes.empty();
  }

  return indices.IsComplete() && extensionsSupported && swapchainAdequate;
}

QueueFamilyIndices Application::FindQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices{};

  uint32_t queueFamilCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilCount, queueFamilies.data());

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphicsFamily = i;

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
    if (presentSupport) indices.presentFamily = i;

    if (indices.IsComplete()) break;

    i++;
  }

  return indices;
}

bool Application::CheckExtensionSupport(VkPhysicalDevice device) {
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());

  std::set<std::string> deviceExtensions(requiredDeviceExtensions.begin(),
                                         requiredDeviceExtensions.end());

  for (const auto &e : availableExtensions) deviceExtensions.erase(e.extensionName);
  return deviceExtensions.empty();
}

SwapchainSupportDetails Application::QuerySwapchainSupportDetails(VkPhysicalDevice device) {
  SwapchainSupportDetails details;

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilites);

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
  if (formatCount != 0) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
  }

  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
  if (presentModeCount != 0) {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                              details.presentModes.data());
  }

  return details;
}

VkSurfaceFormatKHR Application::ChooseSwapchainFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats) {
  for (const auto &format : availableFormats) {
    if (format.format == VK_FORMAT_B8G8R8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      return format;
  }

  return availableFormats[0];
}

VkPresentModeKHR Application::ChooseSwapchainPresentMode(
    const std::vector<VkPresentModeKHR> &availablePresentModes) {
  for (const auto &mode : availablePresentModes) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Application::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilites) {
  if (capabilites.currentExtent.width != UINT32_MAX) return capabilites.currentExtent;

  int width, height;
  SDL_GetWindowSize(window, &width, &height);

  VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

  actualExtent.width = std::clamp(actualExtent.width, capabilites.minImageExtent.width,
                                  capabilites.maxImageExtent.width);
  actualExtent.height = std::clamp(actualExtent.height, capabilites.minImageExtent.height,
                                   capabilites.maxImageExtent.height);

  return actualExtent;
}
