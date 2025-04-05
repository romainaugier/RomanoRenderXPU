#include "romanorender/light.h"
#include "romanorender/sampling.h"

ROMANORENDER_NAMESPACE_BEGIN

/* Square light */

/* https://blogs.autodesk.com/media-and-entertainment/wp-content/uploads/sites/162/egsr2013_spherical_rectangle.pdf */

struct SphericalQuad
{
    Vec3F o;
    Vec3F x, y, z;

    float x0, y0;
    float x1, y1;
    float z0;

    float z0sq;
    float y0sq;
    float y1sq;

    float b0, b1;
    float b0sq;
    float k;
    float S;
};

void spherical_quad_init(SphericalQuad& squad,
                         const Vec3F& s,
                         const Vec3F& ex,
                         const Vec3F& ey,
                         const Vec3F& hit_position) noexcept
{
    squad.o = hit_position;
    const float exl = length_vec3f(ex);
    const float eyl = length_vec3f(ey);

    // compute local reference system ’R’
    squad.x = ex / exl;
    squad.y = ey / eyl;
    squad.z = cross_vec3f(squad.x, squad.y);
    const Vec3F d = s - hit_position;
    squad.z0 = dot_vec3f(d, squad.z);

    // flip ’z’ to make it point against ’Q’
    if(squad.z0 > 0)
    {
        squad.z *= -1;
        squad.z0 *= -1;
    }

    squad.z0sq = squad.z0 * squad.z0;
    squad.x0 = dot_vec3f(d, squad.x);
    squad.y0 = dot_vec3f(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;
    squad.y0sq = squad.y0 * squad.y0;
    squad.y1sq = squad.y1 * squad.y1;

    // create vectors to four vertices
    const Vec3F v00(squad.x0, squad.y0, squad.z0);
    const Vec3F v01(squad.x0, squad.y1, squad.z0);
    const Vec3F v10(squad.x1, squad.y0, squad.z0);
    const Vec3F v11(squad.x1, squad.y1, squad.z0);

    // compute normals to edges
    const Vec3F n0 = normalize_vec3f(cross_vec3f(v00, v10));
    const Vec3F n1 = normalize_vec3f(cross_vec3f(v10, v11));
    const Vec3F n2 = normalize_vec3f(cross_vec3f(v11, v01));
    const Vec3F n3 = normalize_vec3f(cross_vec3f(v01, v00));

    // compute internal angles (gamma_i)
    const float g0 = maths::acosf(-dot_vec3f(n0, n1));
    const float g1 = maths::acosf(-dot_vec3f(n1, n2));
    const float g2 = maths::acosf(-dot_vec3f(n2, n3));
    const float g3 = maths::acosf(-dot_vec3f(n3, n0));

    // compute predefined constants
    squad.b0 = n0.z;
    squad.b1 = n2.z;
    squad.b0sq = squad.b0 * squad.b0;
    squad.k = maths::constants::two_pi - g2 - g3;

    // compute solid angle from internal angles
    squad.S = g0 + g1 + g2 + g3 - maths::constants::two_pi;
}

Vec3F spherical_quad_sample(const SphericalQuad& squad, const Vec2F& sample) noexcept
{
    // 1. compute ’cu’
    const float au = sample.x * squad.S + squad.k;

    float sin_au, cos_au;
    maths::sincosf(au, &sin_au, &cos_au);

    const float fu = (cos_au * squad.b0 - squad.b1) / sin_au;
    float cu = maths::rsqrtf(fu * fu + squad.b0sq) * (fu > 0.0f ? 1.0f : -1.0f);
    cu = maths::clampf(cu, -1, 1);

    // 2. compute ’xu’
    float xu = -(cu * squad.z0) / maths::sqrtf(1 - cu * cu);
    xu = maths::clampf(xu, squad.x0, squad.x1);

    // 3. compute ’yv’
    float d = maths::sqrtf(xu * xu + squad.z0sq);
    float h0 = squad.y0 / maths::sqrtf(d * d + squad.y0sq);
    float h1 = squad.y1 / maths::sqrtf(d * d + squad.y1sq);
    float hv = h0 + sample.y * (h1 - h0), hv2 = hv * hv;
    float yv = (hv2 < 1 - maths::constants::flt_epsilon) ? (hv * d) / maths::sqrtf(1 - hv2) : squad.y1;

    // 4. transform (xu,yv,z0) to world coords
    return (squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z);
}

Vec3F LightSquare::sample_direction_area(const Vec3F& hit_position,
                                         const Vec2F& sample,
                                         const Vec3F& hit_normal,
                                         float& pdf) const noexcept
{
    const Vec3F sample_local((sample.x - 0.5f) * this->_size_x, (sample.y - 0.5f) * this->_size_y, 0.0f);
    const Vec3F sample_world = this->get_transform().transform_point(sample_local);

    const Vec3F direction = normalize_vec3f(sample_world - hit_position);

    const Vec3F light_world_normal = normalize_vec3f(this->get_transform().transform_dir(Vec3F(0, 0, 1)));

    const float area = (this->_size_x * this->_size_y);
    const float cos_theta = dot_vec3f(light_world_normal, direction);
    const float dist2 = dist2_vec3f(hit_position, sample_world);

    pdf = cos_theta < maths::constants::flt_large_epsilon ? 0.0f : (dist2 / (cos_theta * area));

    return direction;
}

Vec3F LightSquare::sample_direction(const Vec3F& hit_position,
                                    const Vec2F& sample,
                                    const Vec3F& hit_normal,
                                    float& pdf) const noexcept
{
    Vec3F center_world;
    this->get_transform().decomp_translation(&center_world);

    const Vec3F ex_world = this->get_transform().transform_dir(Vec3F(this->_size_x, 0.0f, 0.0f));
    const Vec3F ey_world = this->get_transform().transform_dir(Vec3F(0.0f, this->_size_y, 0.0f));

    const Vec3F s_world = center_world - 0.5f * ex_world - 0.5f * ey_world;

    SphericalQuad squad;
    spherical_quad_init(squad, s_world, ex_world, ey_world, hit_position);

    if(squad.S <= maths::constants::flt_epsilon)
    {
        pdf = 0.0f;
        return normalize_vec3f(center_world - hit_position);
    }

    const Vec3F sample_world = spherical_quad_sample(squad, sample);

    const Vec3F direction = sample_world - hit_position;
    const Vec3F light_world_normal = this->get_transform().transform_dir(Vec3F(0.0f, 0.0f, 1.0f));

    // const float cos_theta = dot_vec3f(light_world_normal, direction);
    // pdf = cos_theta < maths::constants::flt_large_epsilon ? 0.0f : 1.0f / (squad.S * cos_theta);
    pdf = 1.0f / squad.S;

    return normalize_vec3f(direction);
}

Vec3F LightSquare::sample_intensity() const noexcept
{
    return this->get_color() * this->get_intensity();
}

/* Dome / Environment light */

Vec3F LightDome::sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal, float& pdf) const noexcept
{
    const Vec3F direction = sample_hemisphere_uniform(sample);
    const float cos_theta = dot_vec3f(direction, hit_normal);

    pdf = sample_hemisphere_uniform_pdf();

    return map_direction_to_normal_vec3f(direction, hit_normal);
}

Vec3F LightDome::sample_intensity() const noexcept
{
    return this->get_color() * this->get_intensity();
}

/* Distant light */

Vec3F LightDistant::sample_direction(const Vec3F& hit_position,
                                     const Vec2F& sample,
                                     const Vec3F& hit_normal,
                                     float& pdf) const noexcept
{
    const Vec2F disk = sample_disk_uniform_concentric(sample);

    const Vec3F sample_3d(disk.x * this->_angle, disk.y * this->_angle, 1.0f);

    pdf = maths::constants::one_over_pi;

    return map_direction_to_normal_vec3f(sample_3d, -this->_orientation);
}

Vec3F LightDistant::sample_intensity() const noexcept
{
    return this->get_color() * this->get_intensity();
}

/* Circle light (same as square light but unit circle instead) */

Vec3F LightCircle::sample_direction(const Vec3F& hit_position,
                                    const Vec2F& sample,
                                    const Vec3F& hit_normal,
                                    float& pdf) const noexcept
{
    const Vec2F disk = sample_disk_uniform_concentric(sample);
    const Vec3F local_pos(disk.x * this->_size_x, disk.y * this->_size_y, 0.0f);

    const Vec3F sample_world = this->get_transform().transform_point(local_pos);

    const Vec3F direction = normalize_vec3f(sample_world - hit_position);

    const Vec3F light_world_normal = normalize_vec3f(this->get_transform().transform_dir(Vec3F(0, 0, 1)));

    const float area = maths::constants::pi * this->_size_x * this->_size_y;
    const float cos_theta = dot_vec3f(light_world_normal, direction);
    const float dist2 = dist2_vec3f(hit_position, sample_world);

    pdf = 1.0f;

    return direction;
}

Vec3F LightCircle::sample_intensity() const noexcept
{
    return this->get_color() * this->get_intensity();
}

/* Spherical light */

Vec3F LightSpherical::sample_direction(const Vec3F& hit_position,
                                       const Vec2F& sample,
                                       const Vec3F& hit_normal,
                                       float& pdf) const noexcept
{
    return Vec3F(0.0f);
}

Vec3F LightSpherical::sample_intensity() const noexcept
{
    return this->get_color() * this->get_intensity();
}

ROMANORENDER_NAMESPACE_END