#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec2 TexCoord;
out vec3 Normal;

uniform mat3 normalMatrix;
uniform mat4 mvpMatrix;
//uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;

void main()
{
	//gl_Position = projection * view * model * vec4(aPos, 1.0f);
	gl_Position = mvpMatrix * vec4(aPos, 1.0f);
	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
	gl_PointSize = 1.0f;
	Normal = normalize(normalMatrix * aNormal);
	Normal.x = Normal.x/2.0f + 0.5f;
	Normal.y = Normal.y/2.0f + 0.5f;
	Normal.z = Normal.z/2.0f + 0.5f;
}