project('dvulkan', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3',
                     'cpp_std=c++20'])

add_project_arguments('-DENABLE_ASSERS', language: 'cpp')

vulkan = dependency('vulkan')
sdl2   = dependency('SDL2')

sources = []

subdir('src')

exe = executable('dvulkan', sources,
                dependencies: [vulkan, sdl2],
                install : true)

test('basic', exe)
