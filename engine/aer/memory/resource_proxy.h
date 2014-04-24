// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_MEMORY_RESOURCE_PROXY_H_
#define AER_MEMORY_RESOURCE_PROXY_H_

#include <string>
#include <unordered_map>

#include "aer/common.h"

// =============================================================================
namespace aer {
// =============================================================================

/**
 * @class ResourceProxy
 * @brief Interface to load and manage resources
 *
 * Implementers must override the 'load' method
*/
template<typename T, typename I=std::string>
class ResourceProxy {
public:
  ResourceProxy() = default;
  virtual ~ResourceProxy();

  /// Load the resource reference by 'id' if not yet in memory and retrieve it
  /// return nullptr if the resource can't be found
  T* get(const I &id);

  /// Release the resource referenced by 'id',
  /// return true if it succeeds.
  bool release(const I &id);

  /// Destroy all the resources still in memory
  void clean_all();

protected:
  typedef std::unordered_map<T*, aer::U32> ProxyRefCountMap_t;
  typedef std::unordered_map<I, T*>        ProxyFileMap_t;

  /// [to override]
  /// Load the resource specified by 'id'
  virtual T* load(const I &id) = 0;

  ProxyRefCountMap_t  reference_counts_;
  ProxyFileMap_t      files_;

private:
  /// increment the reference counter for 'obj'
  void add_reference(T* obj);

  /// Release the resource referenced by 'obj',
  /// @return true if it succeeds.
  /// @note dangerous to use directly (don't void the reference)
  bool release(T* obj);
};

// =============================================================================
}  // namespace aer
// =============================================================================

#include "aer/memory/resource_proxy-inl.h"

#endif  // AER_MEMORY_RESOURCE_PROXY_H_



#if 0
/// ResourceProxy should be used with objects like that to handle reference
/// releasing more nicely.
template<typename T>
class Resource {
 public:
  const T& resource_id() const {
    return resource_id_;
  }

  typedef typename T ResourceType_t;

 private:
  T resource_id_;
};
#endif