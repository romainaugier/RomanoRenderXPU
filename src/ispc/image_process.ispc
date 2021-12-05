export void gammaCorrection(uniform float pixels[], const uniform int size, const uniform float gammaExp)
{
    foreach(i = 0 ... size)
    {
        pixels[i] = pow(pixels[i], gammaExp);
    }
}