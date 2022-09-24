cbuffer WVP : register(b0)
{
    matrix wvp;
};

struct PSInput 
{
    float4 position: SV_POSITION;
    float3 normal: NORMAL;
};

PSInput VSMain(float4 position: POSITION, float3 normal: NORMAL)
{
    PSInput result;
    result.position = mul(wvp, position);
    result.normal = normal;

    return result;
}

float4 PSMain(PSInput input): SV_TARGET
{
    return float4(input.normal, 1.0);
}