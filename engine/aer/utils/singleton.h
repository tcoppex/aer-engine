// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_UTILS_SINGLETON_H_
#define AER_UTILS_SINGLETON_H_

#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Wrapper class to define general purpose Singleton.
///
/// Initialize & Deinitialize are sets to control
/// the construction / destruction orders.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
template <class T>
class Singleton
{
 public:
  static
  void Initialize() {
    AER_ASSERT(sInstance == nullptr);
    sInstance = new T;
  }

  static
  void Deinitialize() {
    AER_CHECK(sInstance != nullptr);
    AER_SAFE_DELETE(sInstance);
  }

  static 
  T& Get() {
    AER_ASSERT(sInstance != nullptr);
    return *sInstance;
  }

 protected:
  Singleton() {}
  virtual ~Singleton() {}

 private:
  static T* sInstance;

  DISALLOW_COPY_AND_ASSIGN(Singleton);
};

template<class T> T* Singleton<T>::sInstance = nullptr;

#if 0
// example
class Logger : public Singleton<Logger> {
  friend class Singleton<Logger>;
  // class specific code
};
#endif

}  // namespace aer

#endif  // AER_UTILS_SINGLETON_H_
