
//
// This source file is part of appleseed.
// Visit http://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2014-2017 Srinath Ravichandran, The appleseedhq Organization
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

// Interface header.
#include "hairbsdf.h"

// appleseed.renderer headers.
#include "renderer/kernel/lighting/scatteringmode.h"
#include "renderer/kernel/shading/directshadingcomponents.h"
#include "renderer/modeling/bsdf/bsdf.h"
#include "renderer/modeling/bsdf/bsdfwrapper.h"

// appleseed.foundation headers.
#include "foundation/math/basis.h"
#include "foundation/math/fresnel.h"
#include "foundation/math/sampling/mappings.h"
#include "foundation/math/scalar.h"
#include "foundation/math/vector.h"
#include "foundation/utility/api/specializedapiarrays.h"
#include "foundation/utility/containers/dictionary.h"

// Standard headers.
#include <cmath>
#include <numeric>

// Forward declarations.
namespace foundation { class IAbortSwitch; }
namespace renderer { class Assembly; }
namespace renderer { class Project; }

using namespace foundation;
using namespace std;

namespace renderer
{

    namespace
    {
        static const int pMax = 3;                      // number of modes within the bsdf we explicitly compute
        static const float SqrtPiOver8 = 0.626657069f;  // a value used in computations within the hairbsdf

        //
        // Some temporary utility functions
        //

        inline float Sqr(float v) { return v * v; }

        template<int n>
        static float Pow(float v)
        {
            static_assert(n > 0, "Power cannot be negative\n");
            float n2 = Pow<n / 2>(v);
            return n2 * n2 * Pow<n & 1>(v);
        }

        template<> float Pow<1>(float v) { return v; }
        template<> float Pow<0>(float v) { return 1; }

        // Safe arc-sin
        inline float safeASin(float val)
        {
            return std::asin(clamp(val, -1.0f, 1.0f));
        }

        // Safe sqrt
        inline float safeSqrt(float val)
        {
            return std::sqrt(std::max(0.0f, val));
        }

        // Modified Bessel functions of the first kind
        // I0(z) = \sum_{i = 0}^{infinity}{((1/4 * z^2)^i)/(i!)^2}
        // I0(z) = \sum_{i = 0}^{infinity}{z^(2i) / (4^i * (i!)^2))}
        // We use the first 10 terms
        inline float I0(float x)
        {
            float val = 0;
            float x2i = 1;
            float ifact = 1;
            int i4 = 1;
            for (int i = 0; i < 10; i++)
            {
                if (i > 1) ifact *= i;
                val += x2i / (i4 * Sqr(ifact));
                x2i *= x * x;
                i4 *= 4;
            }
            return val;
        }

        inline float LogI0(float x)
        {
            if (x > 12)
                return x + 0.5f * (-std::log(2 * M_PI) + std::log(1 / x) + 1 / (8 * x));
            else
                return std::log(I0(x));
        }

        // function that computes the total angle difference
        inline float Phi(int p, float gammaO, float gammaT)
        {
            return 2 * p * gammaT - 2 * gammaO + p * M_PI;
        }

        // Logistic function
        inline float logistic(float x, float s)
        {
            x = std::abs(x);
            return std::exp(-x / s) / (s * Sqr(1 + std::exp(-x / s)));
        }

        // integral of the logistic function
        inline float logisticCDF(float x, float s)
        {
            return 1.0f / (1.0f + std::exp(-x / s));
        }

        // trimmed logistic
        inline float trimmedLogistic(float x, float s, float lb, float ub)
        {
            assert(lb < ub, "the lower bound should be smaller than upper bound");
            return logistic(x, s) / (logisticCDF(ub, s) - logisticCDF(lb, s));
        }

        static float sampleTrimmedLogistic(float u, float s, float a, float b)
        {
            float k = logisticCDF(b, s) - logisticCDF(a, s);
            float x = -s * std::log((1 / (u * k + logisticCDF(a, s))) - 1.0f);
            return clamp(x, a, b);
        }

        // longitudinal scattering
        static float Mp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v)
        {
            float a = cosThetaI * cosThetaO;
            float b = sinThetaI * sinThetaO;
            float mp = (v <= .1f) ?
                (std::exp(LogI0(a) - b - (1.0f / v) + 0.6931f + std::log(1.0f / (2 * v)))) :
                (std::exp(-b) * I0(a)) / (std::sinh(1.0f / v) * 2 * v);
            return mp;
        }

        // Absorption
        static std::array<Spectrum, pMax + 1> Ap(float cosThetaO, float eta, float h, const Spectrum& T)
        {
            std::array<Spectrum, pMax + 1> ret;

            // compute p0 - R
            float cosGammaO = safeSqrt(1.0f - h * h);
            float cosTheta = cosGammaO * cosThetaO;
            float f;
            fresnel_reflectance_dielectric(f, eta, cosTheta);
            ret[0] = Spectrum(f);

            // compute p1 - TT
            ret[1] = Sqr(1.0 - f) * T;

            // compute p2 - TRT
            for (int i = 2; i < pMax; i++)
                ret[i] = ret[i - 1] * T * f;

            // compute all other higher order bounces
            ret[pMax] = ret[pMax - 1] * f * T / (Spectrum(1.0f) - T * f);

            // return the values
            return ret;
        }

        // azimuthal scattering
        inline float Np(float phi, int p, float s, float gammaO, float gammaT)
        {
            float dphi = phi - Phi(p, gammaO, gammaT);

            // remap dphi to be within [-pi, pi]
            while (dphi > M_PI) dphi -= 2 * M_PI;
            while (dphi < -M_PI) dphi += 2 * M_PI;
            return trimmedLogistic(dphi, s, -M_PI, M_PI);
        }

        // utility function to compute fast transmission using exp
        // without rays or anything - just plain distance
        inline Spectrum Exp(const Spectrum& sigma_t, const float distance)
        {
            Spectrum ret;
            for (int i = 0; i < sigma_t.Samples; i++)
                ret[i] = std::exp(-sigma_t[i] * distance);
            return ret;
        }

        // method to compute a discrete pdf based on Ap
        std::array<float, pMax + 1> computeApPdf(float cosThetaO, float eta, float h, const Spectrum& T)
        {
            std::array<float, pMax + 1> ret;
            
            float sinThetaO = safeSqrt(1.0f - Sqr(cosThetaO));
            std::array<Spectrum, pMax + 1> retAp = Ap(cosThetaO, eta, h, T);

            float sumY = std::accumulate(retAp.begin(), retAp.end(), float(0), [](float s, const Spectrum& ap) { return s + luminance(ap); });
            for (int i = 0; i <= pMax; i++)
                ret[i] = luminance(retAp[i]) / sumY;
            return ret;
        }

        // compute the longitudinal variance and azimuthal logistic scale factor
        // v - longitudinal variance factor
        // s - azimuthal logistic scale factor
        // betaM - [0,1] -> v(0) - smooth / v(1) - rough
        void computeAdditionalFactors(float betaM, float betaN, float scaleAlpha, float* v, float& s, float* sin2kAlpha, float* cos2kAlpha) const
        {
            // TODO: We need to find a logic to ask if we can compute the values for multiple lobes
            // greater than pMax = 3
            // even pbrtv3 asks this question
            v[0] = Sqr(0.726f * betaM + 0.812f * Sqr(betaM) + 3.7f * Pow<20>(betaM));
            v[1] = 0.25f * v[0];
            v[2] = 4.0f * v[0];
            for (int p = 3; p <= pMax; p++)
                v[p] = v[2];

            // azimuthal logistic scale factor
            s = SqrtPiOver8 * (0.265f * betaN + 1.194f * Sqr(betaN) + 5.372f * Pow<22>(betaN));

            // compute some sine and cosine terms that account for scale alpha terms
            sin2kAlpha[0] = std::sin(deg_to_rad(scaleAlpha));
            cos2kAlpha[0] = safeSqrt(1.0f - Sqr(sin2kAlpha[0]));
            for (int i = 1; i < pMax; i++)
            {
                sin2kAlpha[i] = 2.0f * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
                cos2kAlpha[i] = Sqr(cos2kAlpha[i - 1]) * Sqr(sin2kAlpha[i - 1]);
            }
        }

        // Utility functions to compute Absorption coefficient based on 
        // parameters

        // Method:1
        // Computes absorption coefficient based on eumelanin and pheomelanin concentrations
        Spectrum sigmaAFromConcentrations(float ce, float cp)
        {
            float sigma_a[3];
            float eumelanin_sigma_a[3] = { 0.419f, 0.697f, 1.37f };
            float pheomelanin_sigma_a[3] = { 0.187f, 0.4f, 1.05f };
            for (int i = 0; i < 3; i++)
                sigma_a[i] = (ce * eumelanin_sigma_a[i] + cp * pheomelanin_sigma_a[i]);
            Spectrum ret;
            linear_rgb_reflectance_to_spectrum_unclamped(Color3f(sigma_a[0], sigma_a[1], sigma_a[2]), ret);
            return ret;
        }

        // Method:2
        // Computes absorption coefficient based on surface roughness and surface reflectance
        Spectrum sigmaAFromReflectance(const Spectrum& reflectance, float beta_n)
        {
            Spectrum sigma_a;
            for (int i = 0; i < Spectrum::Samples; i++)
            {
                sigma_a[i] = Sqr(std::log(reflectance[i]) /
                    (5.969f - 0.215f * beta_n + 2.532f * Sqr(beta_n) -
                    10.73f * Pow<3>(beta_n) + 5.574f * Pow<4>(beta_n) +
                    0.245f * Pow<5>(beta_n)));
            }
            return sigma_a;
        }

        // Utility function to create two random numbers from one random number
        // Based of pbrtv3 implementation
        static uint32_t compactBy1(uint32_t x)
        {
            // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
            x &= 0x55555555;
            // x = --fe --dc --ba --98 --76 --54 --32 --10
            x = (x ^ (x >> 1)) & 0x33333333;
            // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
            x = (x ^ (x >> 2)) & 0x0f0f0f0f;
            // x = ---- ---- fedc ba98 ---- ---- 7654 3210
            x = (x ^ (x >> 4)) & 0x00ff00ff;
            // x = ---- ---- ---- ---- fedc ba98 7654 3210
            x = (x ^ (x >> 8)) & 0x0000ffff;
            return x;
        }

        static Vector2f demuxFloat(float v1)
        {
            uint64_t temp = v1 * (1ull << 32);
            uint32_t bits[2] = { compactBy1(temp), compactBy1(temp >> 1) };
            return Vector2f(bits[0] / (1 << 16), bits[1] / (1 << 16));
        }

        //
        // Hair BSDF.
        //

        const char* Model = "hair_bsdf";

        class HairBSDFImpl
            : public BSDF
        {
        public:
            HairBSDFImpl(
                const char*                 name,
                const ParamArray&           params)
                : BSDF(name, AllBSDFTypes, ScatteringMode::Glossy, params)
            {
                m_inputs.declare("reflectance", InputFormatSpectralReflectance);
                m_inputs.declare("reflectance_multiplier", InputFormatFloat, "1.0");
            }

            void release() override
            {
                delete this;
            }

            const char* get_model() const override
            {
                return Model;
            }

            void sample(
                SamplingContext&            sampling_context,
                const void*                 data,
                const bool                  adjoint,
                const bool                  cosine_mult,
                const int                   modes,
                BSDFSample&                 sample) const override
            {
                if (!ScatteringMode::has_glossy(modes))
                    return;

                // Set the scattering mode.
                sample.m_mode = ScatteringMode::Diffuse;

                // Compute the incoming direction.
                sampling_context.split_in_place(2, 1);
                const Vector2f s = sampling_context.next2<Vector2f>();
                const Vector3f wi = sample_hemisphere_cosine(s);
                sample.m_incoming = Dual3f(sample.m_shading_basis.transform_to_parent(wi));

                // Compute the BRDF value.
                const HairBSDFInputValues* values = static_cast<const HairBSDFInputValues*>(data);
                //sample.m_value.m_diffuse = values->m_reflectance;
                //sample.m_value.m_diffuse *= values->m_reflectance_multiplier * RcpPi<float>();
                sample.m_value.m_beauty = sample.m_value.m_diffuse;

                // Compute the probability density of the sampled direction.
                sample.m_probability = wi.y * RcpPi<float>();
                assert(sample.m_probability > 0.0f);

                sample.compute_reflected_differentials();
            }

            float evaluate(
                const void*                 data,
                const bool                  adjoint,
                const bool                  cosine_mult,
                const Vector3f&             geometric_normal,
                const Basis3f&              shading_basis,
                const Vector3f&             outgoing,
                const Vector3f&             incoming,
                const int                   modes,
                DirectShadingComponents&    value) const override
            {
                // NOTE: What modes do we use here?
                //       to check if we should return 0 or not.
                if (!ScatteringMode::has_glossy(modes))
                    return 0.0f;

                // Compute the BRDF value.
                const HairBSDFInputValues* values = static_cast<const HairBSDFInputValues*>(data);

                // Compute geometric terms
                float sinThetaO = outgoing.x;
                float cosThetaO = safeSqrt(1.0f - Sqr(sinThetaO));
                float phiO = std::atan2(outgoing.z, outgoing.y);
                float gammaO = safeASin(values->m_h);

                float sinThetaI = incoming.x;
                float cosThetaI = safeSqrt(1.0f - Sqr(sinThetaI));
                float phiI = std::atan2(incoming.z, incoming.y);

                // compute terms for the refracted ray
                float eta = values->m_eta;
                float sinThetaT = sinThetaO / eta;
                float cosThetaT = safeSqrt(1.0f - Sqr(sinThetaT));

                // compute the modified refraction coefficient
                // TODO: add the derivation here?
                float etap = std::sqrt(Sqr(eta) - Sqr(sinThetaO)) / cosThetaO;
                float sinGammaT = values->m_h / etap;
                float cosGammaT = safeSqrt(1.0f - Sqr(sinGammaT));
                float gammaT = safeASin(sinGammaT);

                // Compute Attenuation for the single path through the hair cylinder
                Spectrum T = Exp(values->m_sigmaA, 2 * cosGammaT / cosThetaT);

                float phi = phiO - phiI;
                std::array<Spectrum, pMax + 1> ap = Ap(cosThetaO, values->m_h, values->m_eta, T);
                
                // compute the additional terms required for evaluation
                // longitudinal variance factor computed from beta_m
                float v[4];
                float s;
                float sin2kAlpha[3], cos2kAlpha[3];
                computeAdditionalFactors(values->m_betaM, values->m_betaN, values->m_alpha,
                    v, s, sin2kAlpha, cos2kAlpha);
                
                Spectrum fsum;
                // compute the contributions of all the scattering lobes
                for (int p = 0; p < pMax; p++)
                {
                    float sinThetaIp, cosThetaIp;
                    if (p == 0) {
                        sinThetaIp = sinThetaI * cos2kAlpha[1] + cosThetaI * sin2kAlpha[1];
                        cosThetaIp = cosThetaI * cos2kAlpha[1] - sinThetaI * sin2kAlpha[1];
                    }
                    else if (p == 1) {
                        sinThetaIp = sinThetaI * cos2kAlpha[0] - cosThetaI * sin2kAlpha[0];
                        cosThetaIp = cosThetaI * cos2kAlpha[0] + sinThetaI * sin2kAlpha[0];
                    }
                    else if (p == 2) {
                        sinThetaIp = sinThetaI * cos2kAlpha[2] - cosThetaI * sin2kAlpha[2];
                        cosThetaIp = cosThetaI * cos2kAlpha[2] + sinThetaI * sin2kAlpha[2];
                    }
                    else {
                        sinThetaIp = sinThetaI;
                        cosThetaIp = cosThetaI;
                    }

                    // bsdf = Mp * Ap * Np
                    cosThetaIp = std::abs(cosThetaIp);
                    fsum += Mp(cosThetaI, cosThetaO, sinThetaIp, sinThetaO, v[p]) * ap[p] * Np(phi, p, s, gammaO, gammaT);
                }

                // compute the contributions from the rest of the bounces
                fsum += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[pMax]) * ap[pMax] / (2.0f * M_PI);

                // fsum / absCosTheta(Wi)
                fsum /= std::abs(incoming.z);

                value.m_glossy = fsum;
                value.m_beauty = fsum;

                // Return the probability density of the sampled direction.
                return evaluate_pdf(data, adjoint, geometric_normal, shading_basis, outgoing, incoming, modes);
            }

            float evaluate_pdf(
                const void*                 data,
                const bool                  adjoint,
                const Vector3f&             geometric_normal,
                const Basis3f&              shading_basis,
                const Vector3f&             outgoing,
                const Vector3f&             incoming,
                const int                   modes) const override
            {
                if (!ScatteringMode::has_glossy(modes))
                    return 0.0f;

                // Return the probability density of the sampled direction.
                const Vector3f& n = shading_basis.get_normal();
                const float cos_in = abs(dot(incoming, n));
                return cos_in * RcpPi<float>();
            }
        };

        typedef BSDFWrapper<HairBSDFImpl> HairBSDF;
    }


    //
    // HairBSDFFactory class implementation.
    //

    void HairBSDFFactory::release()
    {
        delete this;
    }

    const char* HairBSDFFactory::get_model() const
    {
        return Model;
    }

    Dictionary HairBSDFFactory::get_model_metadata() const
    {
        return
            Dictionary()
            .insert("name", Model)
            .insert("label", "Lambertian BRDF");
    }

    DictionaryArray HairBSDFFactory::get_input_metadata() const
    {
        DictionaryArray metadata;

        metadata.push_back(
            Dictionary()
            .insert("name", "reflectance")
            .insert("label", "Reflectance")
            .insert("type", "colormap")
            .insert("entity_types",
                Dictionary()
                .insert("color", "Colors")
                .insert("texture_instance", "Textures"))
            .insert("use", "required")
            .insert("default", "0.5"));

        metadata.push_back(
            Dictionary()
            .insert("name", "reflectance_multiplier")
            .insert("label", "Reflectance Multiplier")
            .insert("type", "colormap")
            .insert("entity_types",
                Dictionary().insert("texture_instance", "Textures"))
            .insert("use", "optional")
            .insert("default", "1.0"));

        return metadata;
    }

    auto_release_ptr<BSDF> HairBSDFFactory::create(
        const char*         name,
        const ParamArray&   params) const
    {
        return auto_release_ptr<BSDF>(new HairBSDF(name, params));
    }

}   // namespace renderer
