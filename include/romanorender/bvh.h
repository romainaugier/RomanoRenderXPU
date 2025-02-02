#pragma once

#include "sphere.h"
#include <vector>
#include <algorithm>
#include <atomic>


struct Node
{
	BoundingBox bbox;

	Node* left = nullptr;
	Node* right = nullptr;

	uint32_t primOffset = 0;
	uint8_t axis = 0;
	uint8_t primCount = 0;

	Node() {}

	Node(const BoundingBox& bbox, 
		 Node* left, 
		 Node* right, 
		 const uint32_t offset, 
		 const uint8_t axis, 
		 const uint8_t count) :
		bbox(bbox),
		left(left),
		right(right),
		primOffset(offset),
		axis(axis),
		primCount(count)
	{}
};

struct alignas(32) LinearNode
{
	BoundingBox bbox;
	
	union {
		uint32_t secondChildOffset = 0;
		uint32_t primitivesOffset;
	};

	uint16_t level = 0;

	uint8_t axis = 0;
	uint8_t primCount = 0;
};

using nodeAllocator = tbb::cache_aligned_allocator<LinearNode>;

struct Allocators
{	
	nodeAllocator tbbNodeAllocator;

	spheresNAllocator tbbSpheresAllocator;

	floatAllocator tbbFloat32Allocator;
	intAllocator tbbUInt32Allocator;

};

struct Accelerator
{
	SpheresN* spheres = nullptr;

	LinearNode* linearBvhHead = nullptr;

	uint32_t nodeCount = 0;
	uint32_t spheresCount = 0;
};

Accelerator BuildAccelerator(const std::vector<Sphere>& sphere, Allocators& allocators) noexcept;

bool Intersect(const Accelerator& accelerator, 
			   RayHit& rayhit) noexcept;

bool Occlude(const Accelerator& accelerator, 
			 ShadowRay& shadow) noexcept;

void ReleaseAcceleratorTemp(Node* bvhHead) noexcept;

void ReleaseAccelerator(Accelerator& accelerator, Allocators& allocators) noexcept;

struct PrimitiveInfo
{
	BoundingBox bbox;

	vec3 centroid;

	uint32_t index;

	PrimitiveInfo(const BoundingBox& bbox, const vec3& center, const uint32_t idx) :
		bbox(bbox),
		centroid(center),
		index(idx) {}
};

struct Bucket
{
	BoundingBox bbox;

	uint32_t count = 0;
};

Node* BuildNode(const std::vector<Sphere>& spheres, 
				std::vector<PrimitiveInfo>& primInfos, 
				std::vector<Sphere>& orderedSpheres, 
				uint32_t start, 
				uint32_t end, 
				uint32_t& nodeCount) noexcept;

void DeleteNode(Node* node) noexcept;

uint32_t FlattenAccelerator(LinearNode* linearBvh, 
							Node* node, 
							uint32_t* offset) noexcept;