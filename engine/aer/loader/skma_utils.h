// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_SKMA_UTILS_H_
#define AER_LOADER_SKMA_UTILS_H_

#include <vector>
#include "aer/common.h"

namespace aer {

class SKMFile;

namespace skmautils {

/// Compute mesh normals from faces
void ComputeNormals(const SKMFile &skmFile, 
                    std::vector<aer::Vector3> &normals);

/// Gather joint datas into single buffers
void SetupJointsData(const SKMFile &skmFile, 
                     std::vector<aer::Vector4i> &indices,
                     std::vector<aer::Vector3>  &weights);

}  // namespace skmautils
}  // namespace aer

#endif  // AER_LOADER_SKMA_UTILS_H_