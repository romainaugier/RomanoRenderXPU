__forceinline__ __device__ float rng(unsigned int& previous)
{
    previous = previous * 1664525u + 1013904223u;

    return float(previous & 0x00FFFFFF) / float(0x01000000u);
}