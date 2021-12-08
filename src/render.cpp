#include "render.h"

void GenerateTiles(Tiles& tiles, 
                   const Settings& settings) noexcept
{   
    constexpr uint8_t tilesize = 16;

    const uint16_t tileCountX = ceil(settings.xres / tilesize);
    const uint16_t tileCountY = ceil(settings.yres / tilesize);

    const uint8_t lastTileSizeX = ceil(fmodf(settings.xres, tilesize)) == 0 ? 32 : ceil(fmodf(settings.xres, tilesize));
    const uint8_t lastTileSizeY = ceil(fmodf(settings.yres, tilesize)) == 0 ? 32 : ceil(fmodf(settings.yres, tilesize));

    tiles.count = tileCountX * tileCountY;

    tiles.tiles.reserve(tiles.count);

    uint16_t idx = 0;

    for (int y = 0; y < settings.yres; y += tilesize)
    {
        for (int x = 0; x < settings.xres; x += tilesize)
        {
            Tile tmpTile;
            tmpTile.id = idx;
            tmpTile.x_start = x; tmpTile.x_end = x + tilesize;
            tmpTile.y_start = y; tmpTile.y_end = y + tilesize;

            if (x + tilesize > settings.xres)
            {
                tmpTile.x_end = x + lastTileSizeX;
                tmpTile.size_x = lastTileSizeX;
            }
            if (y + tilesize > settings.yres)
            {
                tmpTile.y_end = y + lastTileSizeY;
                tmpTile.size_y = lastTileSizeY;
            }

            tmpTile.pixels = new color[tmpTile.size_x * tmpTile.size_y];
            tmpTile.randoms = new float[tmpTile.size_x * tmpTile.size_y * 2];

            tiles.tiles.push_back(tmpTile);

            idx++;
        }
    }
}

void ReleaseTiles(Tiles& tiles) noexcept
{
    for (auto& tile : tiles.tiles)
    {
        delete[] tile.pixels;
        tile.pixels = nullptr;

        delete[] tile.randoms;
        tile.randoms = nullptr;
    }
}

void SetTilePixel(Tile& tile, const vec3& color, uint32_t x, uint32_t y) noexcept
{
    tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].R = color.x;
    tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].G = color.y;
    tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].B = color.z;
}

void Render(color* __restrict buffer,
            const Accelerator& accelerator,
            const std::vector<Material*>& materials,
            const uint32_t* blueNoise,
            const uint64_t& seed,
            const uint64_t& sample,
            const Tiles& tiles, 
            const Camera& cam, 
            const Settings& settings) noexcept
{
    static tbb::affinity_partitioner partitioner;

    const float gamma = 1.0f / settings.gamma;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, tiles.count), [&](const tbb::blocked_range<size_t>& r)
        {
            for (size_t t = r.begin(), t_end = r.end(); t < t_end; t++)
            {
                RenderTile(accelerator, materials, blueNoise, seed, sample, tiles.tiles[t], cam, settings);

                for (int y = 0; y < tiles.tiles[t].size_y; y++)
                {
                    for (int x = 0; x < tiles.tiles[t].size_x; x++)
                    {
                        buffer[tiles.tiles[t].x_start + x + (tiles.tiles[t].y_start + y) * settings.xres].R = tiles.tiles[t].pixels[x + y * tiles.tiles[t].size_x].R;
                        buffer[tiles.tiles[t].x_start + x + (tiles.tiles[t].y_start + y) * settings.xres].G = tiles.tiles[t].pixels[x + y * tiles.tiles[t].size_x].G;
                        buffer[tiles.tiles[t].x_start + x + (tiles.tiles[t].y_start + y) * settings.xres].B = tiles.tiles[t].pixels[x + y * tiles.tiles[t].size_x].B;
                    }
                }
                
                for(int y = 0; y < tiles.tiles[t].size_y; y++)
                    ispc::gammaCorrection((float*)&buffer[tiles.tiles[t].x_start + (tiles.tiles[t].y_start + y) * settings.xres].R, tiles.tiles[t].size_x * 3, gamma);
            }

        }, partitioner);
}

void RenderTile(const Accelerator& accelerator,
                const std::vector<Material*>& materials,
                const uint32_t* blueNoise,
                const uint64_t& seed,
                const uint64_t& sample,
                const Tile& tile,
                const Camera& cam,
                const Settings& settings) noexcept
{
    // Method to see if we intersect something in the tile by tracing a ray at each corner. If we don't, just 
    // sample the background and do another tile
    // Not very robust, but it will get somewhere
    uint32_t pos2d[8] = { tile.x_start, tile.y_start, tile.x_end - 1, tile.y_start,
                          tile.x_start, tile.y_end - 1, tile.x_end, tile.y_end - 1 };

    bool hasHitSomething = false;

    for(uint8_t i = 0; i < 4; i++)
    {
        RayHit tmpRayHit;

        SetPrimaryRay(tmpRayHit, cam, pos2d[i * 2], pos2d[i * 2 + 1], settings.xres, settings.yres, blueNoise, sample);

        if(Intersect(accelerator, tmpRayHit))
        {
            hasHitSomething = true;
            break;
        }
    }

    if(hasHitSomething)
    {
        for (int y = tile.y_start; y < tile.y_end; y++)
        {
            for (int x = tile.x_start; x < tile.x_end; x++)
            {
                RayHit tmpRayHit;

                SetPrimaryRay(tmpRayHit, cam, x, y, settings.xres, settings.yres, blueNoise, sample);
                
                const vec3 output = Pathtrace(accelerator, materials, blueNoise, x, y, sample, seed * 9483 * x * y, tmpRayHit);

                const vec3 outputCorrected = vec3(std::isnan(output.x) ? 0.5f : output.x, 
                                                std::isnan(output.y) ? 0.5f : output.y, 
                                                std::isnan(output.z) ? 0.5f : output.z);

                const float pixelR = tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].R;
                const float pixelG = tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].G;
                const float pixelB = tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].B;

                tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].R = lerp(pixelR, outputCorrected.x, 1.0f / static_cast<float>(sample));
                tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].G = lerp(pixelG, outputCorrected.y, 1.0f / static_cast<float>(sample));
                tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].B = lerp(pixelB, outputCorrected.z, 1.0f / static_cast<float>(sample));
            }
        }
    }
    else
    {
        for (int y = tile.y_start; y < tile.y_end; y++)
        {
            for (int x = tile.x_start; x < tile.x_end; x++)
            {
                RayHit tmpRayHit;

                SetPrimaryRay(tmpRayHit, cam, x, y, settings.xres, settings.yres, blueNoise, sample);
                
                const vec3 output = lerp(vec3(0.3f, 0.5f, 0.7f), vec3(1.0f), fit(tmpRayHit.ray.direction.y, -1.0f, 1.0f, 0.0f, 1.0f));

                const float pixelR = tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].R;
                const float pixelG = tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].G;
                const float pixelB = tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].B;

                tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].R = lerp(pixelR, output.x, 1.0f / static_cast<float>(sample));
                tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].G = lerp(pixelG, output.y, 1.0f / static_cast<float>(sample));
                tile.pixels[(x - tile.x_start) + (y - tile.y_start) * tile.size_y].B = lerp(pixelB, output.z, 1.0f / static_cast<float>(sample));
            }
        }
    }
}

vec3 Pathtrace(const Accelerator& accelerator,
               const std::vector<Material*>& materials,
               const uint32_t* blueNoise,
               const uint32_t x,
               const uint32_t y,
			   const uint32_t sample,
               const uint32_t seed, 
               RayHit& rayhit) noexcept
{
    vec3 output(0.0f);
    vec3 weight(1.0f);

    // constexpr unsigned int floatAddr = 0x2f800004u;
    // auto toFloat = float();
    // memcpy(&toFloat, &floatAddr, 4);

    if (Intersect(accelerator, rayhit))
    {
        vec3 hitPosition = rayhit.hit.pos;
        vec3 hitNormal = rayhit.hit.normal;

        uint32_t hitId = rayhit.hit.geomID;
        uint32_t hitMatId = rayhit.hit.matID;

        uint8_t bounce = 0;

        while (true)
        {
            if (bounce > 6)
            {
                break;
            }

            const float rx = BlueNoiseSamplerSpp(blueNoise, x, y, sample, 2);
            const float ry = BlueNoiseSamplerSpp(blueNoise, x, y, sample, 3);
            
            // Sample Light
            float lightIntensity = 1.0f;
            ShadowRay shadow;
            
            shadow.origin = hitPosition + hitNormal * 0.001f;
            shadow.direction = materials[hitMatId]->Sample(hitNormal, rayhit.ray.direction, rx, ry);
            shadow.inverseDirection = 1.0f / shadow.direction;
            shadow.t = 1000000.0f;
            
            if(materials[hitMatId]->m_Type & MaterialType_Diffuse)
            {            
                if (Occlude(accelerator, shadow))
                {
                    lightIntensity = 0.0f;
                }
            }

            const vec3 hitColor = materials[hitMatId]->Eval(hitNormal, shadow.direction) * lightIntensity * weight;
            weight *= hitColor;

            output += materials[hitMatId]->m_Type & MaterialType_Diffuse ? hitColor : 0.0f;

            const float rr = min(0.95f, (0.2126 * weight.x + 0.7152 * weight.y + 0.0722 * weight.z));
            if (rr < rx) break;
            else weight /= rr;

            SetRay(rayhit, hitPosition, materials[hitMatId]->Sample(hitNormal, rayhit.ray.direction, rx, ry), 10000.0f);

            if(rayhit.ray.direction == hitNormal) break;

            if (!Intersect(accelerator, rayhit))
            {
                // Sky Color
                if(materials[hitMatId]->m_Type & MaterialType_Reflective || materials[hitMatId]->m_Type & MaterialType_Dielectric)
                {
                    output = lerp(vec3(0.3f, 0.5f, 0.7f), vec3(1.0f), fit(rayhit.ray.direction.y, -1.0f, 1.0f, 0.0f, 1.0f)) * materials[hitMatId]->m_Color;
                }

                break;
            }

            hitPosition = rayhit.hit.pos;
            hitNormal = rayhit.hit.normal;
            hitId = rayhit.hit.geomID;
            hitMatId = rayhit.hit.matID;

            bounce++;
        }
    }
    else
    {
        output = lerp(vec3(0.3f, 0.5f, 0.7f), vec3(1.0f), fit(rayhit.ray.direction.y, -1.0f, 1.0f, 0.0f, 1.0f));
    }

    return output;
}
