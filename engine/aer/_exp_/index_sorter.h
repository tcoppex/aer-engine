// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_INDEX_SORTER_H_
#define AER_INDEX_SORTER_H_

#include <algorithm>
#include <vector>
#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Creates a buffer of sorted indices along an arbitrary
/// axis for a fiven buffer of coordinates.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class IndexSorter {
  public:
    IndexSorter() {
      indices_.clear();
      attribs_.clear();
    }

    /// TODO : accept buffer of const Position3D to sort

    /// Generic sorting of an attrib j of any N-vector buffer (j < N)
    template<U32 nDim>
    void sort_dim(const U32 nelems, const U32 dim, const F32 *data) {
      AER_ASSERT((dim >= 0u) && (dim < nDim));

      reset_indices(nelems);
      reset_attribs<nDim>(nelems, dim, data);

      sort();
    }

    void sort_x(const U32 nelems, const F32 *data, U32 *dst = nullptr) { 
      sort_dim<3>(nelems, 0, data);

      if (dst != nullptr) {
        copy(dst);
      }
    }

    void sort_y(const U32 nelems, const F32 *data, U32 *dst = nullptr) { 
      sort_dim<3>(nelems, 1, data); 

      if (dst != nullptr) {
        copy(dst);
      }
    }

    void sort_z(const U32 nelems, const F32 *data, U32 *dst = nullptr) { 
      sort_dim<3>(nelems, 2, data); 

      if (dst != nullptr) {
        copy(dst);
      }
    }

    void sort_axis(const Vector3 &axis, U32 nelems, const F32 *data, U32 *dst = nullptr) {
      reset_indices(nelems);

      // Normalize axis (not really needed)
      Vector3 sorted_axis = glm::normalize(axis);
      reset_attribs(nelems, sorted_axis, data);

      sort();

      if (dst != nullptr) {
        copy(dst);
      }
    }

    void copy(U32 *dst) {
      std::copy(indices_.begin(), indices_.end(), dst);
    }

    U32 size() const { return indices_.size(); }
    U32 index(const U32 idx) const { return indices_[idx]; }
    const U32* indices() const { return indices_.data(); }


  private:
    /// Base struct functor to sort by LOWER values
    struct SortL {
      SortL(const F32 *attrib) : 
        attrib(attrib) 
      {}

      bool operator() (U32 i1, U32 i2) { 
        return attrib[i1] < attrib[i2];
      }

      const F32 *attrib;
    };

    ///
    void sort() {
      std::sort(indices_.begin(), indices_.end(), SortL(attribs_.data()));
      attribs_.clear(); //
    }

    ///
    void reset_indices(U32 nelems) {
      indices_.resize(nelems);
      for (U32 i = 0u; i < nelems; ++i) {
        indices_[i] = i;
      }
    }

    ///
    template<U32 nDim>
    void reset_attribs(U32 nelems, U8 dim, const F32 *data) {
      attribs_.resize(nelems);
      for (U32 i = 0u; i < nelems; ++i) {
        attribs_[i] = data[i*nDim + dim];
      }
    }
    
    ///
    void reset_attribs(U32 nelems, const Vector3 &axis, const F32 *data) {
      attribs_.resize(nelems);
      for (U32 i = 0u; i < nelems; ++i) {
        const F32 *v = &data[3u*i];
        attribs_[i] = v[0]*axis[0] + v[1]*axis[1] + v[2]*axis[2];
      }
    }


    std::vector<U32> indices_;
    std::vector<F32> attribs_;             // Values used to sorted datas
};

} // aer

#endif  // AER_INDEX_SORTER_H_
