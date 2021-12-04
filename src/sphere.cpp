#include "sphere.h"

void AllocateSphereN(SpheresN& spheres, 
					 const size_t spheresCount, 
					 floatAllocator* fAllocator, 
					 intAllocator* iAllocator) noexcept
{
	spheres.centerX = fAllocator->allocate(spheresCount);	
	spheres.centerY = fAllocator->allocate(spheresCount);	
	spheres.centerZ = fAllocator->allocate(spheresCount);	

	spheres.radius = fAllocator->allocate(spheresCount);

	spheres.id = iAllocator->allocate(spheresCount);
	spheres.matId = iAllocator->allocate(spheresCount);
}

void ReleaseSphereN(SpheresN& spheres,
					const size_t spheresCount,
					floatAllocator* fAllocator, 
					intAllocator* iAllocator) noexcept
{
	fAllocator->deallocate(spheres.centerX, spheresCount);	
	fAllocator->destroy(spheres.centerX);	

	fAllocator->deallocate(spheres.centerY, spheresCount);	
	fAllocator->destroy(spheres.centerY);

	fAllocator->deallocate(spheres.centerZ, spheresCount);
	fAllocator->destroy(spheres.centerZ);

	fAllocator->deallocate(spheres.radius, spheresCount);
	fAllocator->destroy(spheres.radius);

	iAllocator->deallocate(spheres.id, spheresCount);
	iAllocator->destroy(spheres.id);

	iAllocator->deallocate(spheres.matId, spheresCount);	
	iAllocator->destroy(spheres.matId);	
}

bool SphereHit(const Sphere& sphere, RayHit& rayhit) noexcept
{
	const vec3 oc = rayhit.ray.origin - sphere.center;
	const auto a = length2(rayhit.ray.direction);
	const auto b = dot(oc, rayhit.ray.direction);
	const auto c = length2(oc) - sphere.radius * sphere.radius;
	const auto discriminant = b * b - a * c;
	
	if (discriminant > 0)
	{
		const float t = (-b - sqrtf(discriminant)) / a;

		if (t > 0.0f && t < rayhit.ray.t)
		{
			rayhit.ray.t = t;
			rayhit.hit.pos = rayhit.ray.origin + (rayhit.ray.direction * t);
			rayhit.hit.normal = (rayhit.hit.pos - sphere.center) / sphere.radius;
			rayhit.hit.geomID = sphere.id;
			rayhit.hit.matID = sphere.matId;

			return true;
		}
	}
	
	return false;
}

bool SphereHitN(const SpheresN& spheres, RayHit& rayhit, const int start, const int count) noexcept
{
	bool hit = false;

	float t[8];
	for(int i = 0; i < count; i++) t[i] = rayhit.ray.t;

	ispc::SphereHitN(spheres.centerX, 
					 spheres.centerY, 
					 spheres.centerZ, 
					 spheres.radius, 
					 ToIspcVec(rayhit.ray.origin), 
					 ToIspcVec(rayhit.ray.direction), 
					 t, count, start);

	int8_t idx = -1;

	float tmpT = std::min(rayhit.ray.t, 10000000.0f);

	for (int i = 0; i < count; i++)
	{
		if (t[i] < tmpT)
		{
			tmpT = t[i];
			idx = i;
		}
	}

	if (idx >= 0)
	{
		hit = true;
		rayhit.ray.t = tmpT;
		rayhit.hit.pos = rayhit.ray.origin + (rayhit.ray.direction * tmpT);
		rayhit.hit.normal = (rayhit.hit.pos - vec3(spheres.centerX[start + idx], spheres.centerY[start + idx], spheres.centerZ[start + idx])) / spheres.radius[start + idx];
		rayhit.hit.geomID = spheres.id[start + idx];
		rayhit.hit.matID = spheres.matId[start + idx];
	}

	return hit;
}

bool SphereOcclude(const Sphere& sphere, ShadowRay& shadow) noexcept
{
	const vec3 oc = shadow.origin - sphere.center;
	const auto a = length2(shadow.direction);
	const auto b = dot(oc, shadow.direction);
	const auto c = length2(oc) - sphere.radius * sphere.radius;
	const auto discriminant = b * b - a * c;
	
	if (discriminant > 0)
	{
		const float t = (-b - sqrtf(discriminant)) / a;

		if (t > 0.0f)
		{
			shadow.t = t;
			return true;
		}
	}

	return false;
}

bool SphereOccludeN(const SpheresN& spheres, ShadowRay& shadow, const int start, const int count) noexcept
{
	float t[8];
	for(int i = 0; i < count; i++) t[i] = shadow.t;

	ispc::SphereHitN(spheres.centerX, 
					 spheres.centerY, 
					 spheres.centerZ, 
					 spheres.radius, 
					 ToIspcVec(shadow.origin), 
					 ToIspcVec(shadow.direction), 
					 t, count, start);

	for (int i = 0; i < count; i++)
	{
		if (t[i] < shadow.t)
		{
			shadow.t = t[i];
			return true;
		}
	}

	return false;
}

BoundingBox SphereBBox(const Sphere& sphere) noexcept
{
	return BoundingBox(vec3(sphere.center - sphere.radius), vec3(sphere.center + sphere.radius));
}