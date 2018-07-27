
//
// This source file is part of appleseed.
// Visit https://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2018 Francois Beaune, The appleseedhq Organization
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

#ifndef APPLESEED_RENDERER_MODELING_POSTPROCESSINGSTAGE_COLORMAPPOSTPROCESSINGSTAGE_H
#define APPLESEED_RENDERER_MODELING_POSTPROCESSINGSTAGE_COLORMAPPOSTPROCESSINGSTAGE_H

// appleseed.renderer headers.
#include "renderer/modeling/postprocessingstage/ipostprocessingstagefactory.h"

// appleseed.main headers.
#include "main/dllsymbol.h"

// Forward declarations.
namespace foundation    { class Dictionary; }
namespace foundation    { class DictionaryArray; }
namespace renderer      { class ParamArray; }

namespace renderer
{

//
// A post-processing stage that remaps frame colors using a given color map.
// Typically used to convert a frame to false colors.
//

class APPLESEED_DLLSYMBOL ColorMapPostProcessingStageFactory
  : public IPostProcessingStageFactory
{
  public:
    // Delete this instance.
    void release() override;

    // Return a string identifying this stage model.
    const char* get_model() const override;

    // Return metadata for this stage model.
    foundation::Dictionary get_model_metadata() const override;

    // Return metadata for the inputs of this stage model.
    foundation::DictionaryArray get_input_metadata() const override;

    // Create a new stage instance.
    foundation::auto_release_ptr<PostProcessingStage> create(
        const char*         name,
        const ParamArray&   params) const override;
};

}       // namespace renderer

#endif  // !APPLESEED_RENDERER_MODELING_POSTPROCESSINGSTAGE_COLORMAPPOSTPROCESSINGSTAGE_H
