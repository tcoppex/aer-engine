// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_BLEND_SHAPE_H_
#define AER_ANIMATION_BLEND_SHAPE_H_

#include <map>
#include <string>
#include <vector>

#include "aer/common.h"
#include "aer/animation/common.h"
#include "aer/device/texture_buffer.h"
#include "aer/utils/global_clock.h"

// =============================================================================
namespace aer {
// =============================================================================

class SKMFile;

/**
 * @class BlendShape
 * @brief Contains the blend shapes data used by a base mesh
*/
class BlendShape {
public:
  /// Type of buffers sent to the device.
  ///
  /// Important note :
  ///  - Indices & Weights are per-instance updatable buffers [not use here], 
  ///    but could be shared application-wised if datas are changed every frames.
  ///
  ///  - Data & LUT are shared mesh-wised static buffers.
  enum BSBufferType {
    BS_INDICES,       // indices of blend shape to use
    BS_WEIGHTS,       // weights for each blend shape
    BS_DATAS,         // morph vector for each vertices for each blend shapes
    BS_LUT,           // LUT to retrieve data from vertexID and blendShapeID
    
    kNumBufferType
  };

  BlendShape();
  ~BlendShape();

  void init(const SKMFile &skmFile);

  /// @return a blend shape ID from its name
  U32 id_from_name(const std::string &name) {
    return mBSIndexMap[name];
  }

  /// @return the number of blend shape stored
  U32 count() const {
    return mCount;
  }

  // ----------------------------------------------------------
  /// Bind the given texture buffer
  void bind_texture_buffer(BSBufferType buffer_id, I32 texture_unit) {
    mTBO[buffer_id].texture.bind(texture_unit);
  }

  TBO_t& texture_buffer(BSBufferType buffer_id) {
    return mTBO[buffer_id];
  }
  
  void generate_sequence(Sequence_t &sequence) {
    F32 global_time = GlobalClock::Get().application_time(SECOND);
    sequence.resize(mExpressions.size());
    for (U32 i = 0u; i < sequence.size(); ++i) {
      sequence[i].action_ptr   = &mExpressions[i];
      sequence[i].global_start = global_time;
    }
  }

  void add_expressions(Sequence_t &sequence,
                       Expression_t *expressions,
                       U32 size) {
    mExpressions.insert(mExpressions.begin(), expressions, expressions + size);
    generate_sequence(sequence);
  }  
  // ----------------------------------------------------------

  /// Display the list of blend shapes with their ids
  void DEBUG_display_names();


private:
  void init_expressions();

  typedef std::map<std::string, U32>  BSIndexMap_t;

  /// Individual blendshape index
  BSIndexMap_t mBSIndexMap;
  U32          mCount;

  /// Set of expressions
  std::vector<Expression_t> mExpressions;

  /// Device buffers use for rendering
  TBO_t mTBO[kNumBufferType];
};

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // AER_ANIMATION_BLEND_SHAPE_H_
