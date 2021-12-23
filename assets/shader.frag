#version 450

layout(location = 0) out vec4 color;

layout(location = 0) in vec3 v_Color;

void main() { color = vec4(v_Color, 1.0); }
