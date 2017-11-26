
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
#include "foundation/math/sampling/mappings.h"
#include "foundation/math/scalar.h"
#include "foundation/math/vector.h"
#include "foundation/utility/api/specializedapiarrays.h"
#include "foundation/utility/containers/dictionary.h"

// Standard headers.
#include <cmath>

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

        //
        // Lambertian BRDF.
        //

        const char* Model = "hair_bsdf";

        class HairBSDFImpl
            : public BSDF
        {
        public:
            HairBSDFImpl(
                const char*                 name,
                const ParamArray&           params)
                : BSDF(name, Reflective, ScatteringMode::Glossy | ScatteringMode::Specular, params)
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
                if (!ScatteringMode::has_diffuse(modes))
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
                sample.m_value.m_diffuse = values->m_reflectance;
                sample.m_value.m_diffuse *= values->m_reflectance_multiplier * RcpPi<float>();
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
                if (!ScatteringMode::has_diffuse(modes))
                    return 0.0f;

                // Compute geometric terms
                float sinThetaWo = outgoing.x;
                float cosThetaWo = safeSqrt(1.0f - Sqr(sinThetaWo));
                float phiWo = std::atan2(outgoing.z, outgoing.y);

                float sinThetaWi = incoming.x;
                float cosThetaWi = safeSqrt(1.0f - Sqr(sinThetaWi));
                float phiWi = std::atan2(incoming.z, incoming.y);

                float v = 0.0f;

                // Compute longitudinal scattering
                float Mp = this->Mp(cosThetaWi, cosThetaWo, sinThetaWi, sinThetaWo, v);



                // Compute Attenuation

                // Compute Azimuthal scattering

                // Compute the final BSDF value



                // Compute the BRDF value.
                const HairBSDFInputValues* values = static_cast<const HairBSDFInputValues*>(data);
                value.m_diffuse = values->m_reflectance;
                value.m_diffuse *= values->m_reflectance_multiplier * RcpPi<float>();
                value.m_beauty = value.m_diffuse;

                // Return the probability density of the sampled direction.
                const Vector3f& n = shading_basis.get_normal();
                const float cos_in = abs(dot(incoming, n));
                return cos_in * RcpPi<float>();
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
                if (!ScatteringMode::has_diffuse(modes))
                    return 0.0f;

                // Return the probability density of the sampled direction.
                const Vector3f& n = shading_basis.get_normal();
                const float cos_in = abs(dot(incoming, n));
                return cos_in * RcpPi<float>();
            }
        
        private:

            static float Mp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v)
            {
                float a = cosThetaI * cosThetaO;
                float b = sinThetaI * sinThetaO;
                float mp = (v <= .1f) ?
                    (std::exp(LogI0(a) - b - (1.0f / v) + 0.6931f + std::log(1.0f / (2 * v)))) :
                    (std::exp(-b) * I0(a)) / (std::sinh(1.0f / v) * 2 * v);
                return mp;
            }

            static float Ap(float cosTheta0, float sinTheta0, float h, float eta)
            {
                float sinThetaT = sinTheta0 / eta;
                float cosThetaT = safeSqrt(1.0f - Sqr(sinThetaT));

                float etap = std::sqrt(Sqr(eta) - Sqr(sinTheta0)) / cosTheta0;
                float sinGammaT = h / etap;
                float cosGammaT = safeSqrt(1.0f - Sqr(sinGammaT));
                float gammaT = safeASin(sinGammaT);

                // compute the distance of traverse within the hair
                float l = 2 * cosGammaT / cosThetaT;

                // TODO: compute this for all channels
                float sigma_a;
                float T = exp(-sigma_a * l);
            }

            static float I0(float x)
            {

            }
            
            static float LogI0(float x)
            {

            }

            static const int pMax = 3;      // number of modes within the bsdf we explicitly compute
        };

        typedef BSDFWrapper<HairBSDFImpl> HairBSDF;
    }


    //
    // LambertianBRDFFactory class implementation.
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
