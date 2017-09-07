// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_MEMORY_RESOURCE_PROXY_INL_H_
#define AER_MEMORY_RESOURCE_PROXY_INL_H_

// =============================================================================
namespace aer {
// =============================================================================

template<typename T, typename I>
ResourceProxy<T,I>::~ResourceProxy() {
  clean_all();
}

// -----------------------------------------------------------------------------

template<typename T, typename I>
T* ResourceProxy<T,I>::get(const I &id) {
  T* obj = files_[id];

  if (!obj) {
    obj = load(id);
    files_[id] = obj;
    reference_counts_[obj] = 0u;
  }

  if (obj) { 
    add_reference(obj);
  } 

  return obj;
}

// -----------------------------------------------------------------------------

template<typename T, typename I>
bool ResourceProxy<T,I>::release(T *obj) {
  if (reference_counts_[obj] == 0u) {
    return false;
  }
  
  if (0u == --reference_counts_[obj]) {
    AER_SAFE_DELETE(obj);
  }
  return true;
}

// -----------------------------------------------------------------------------

template<typename T, typename I>
bool ResourceProxy<T,I>::release(const I &id) {
  typename ProxyFileMap_t::iterator it = files_.find(id);
  if (it == files_.end()) {
    return false;
  }
  
  if (release(it->second)) {
    files_[it->first] = nullptr;
    return true;
  }

  return false;
}

// -----------------------------------------------------------------------------

template<typename T, typename I>
void ResourceProxy<T,I>::clean_all() {
  for (auto &file : files_) {
    AER_SAFE_DELETE(file.second);
    files_[file.first] = nullptr;
  }
}

// -----------------------------------------------------------------------------

template<typename T, typename I>
void ResourceProxy<T,I>::add_reference(T* obj) {
  ++reference_counts_[obj]; 
}

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // AER_MEMORY_RESOURCE_PROXY_INL_H_
