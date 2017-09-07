// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_UTILS_PATH_H_
#define AER_UTILS_PATH_H_

#include <string>
#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Path {
 private:
  std::string canonical_;
  std::string canonical_directory_; //
  U32 name_idx_;


 public:
  /// Clean a filename from special token (eg. '.', '..', '//')
  static
  void CleanFilename(std::string &filename);

  explicit Path(const std::string &pathname);
  Path(const Path &path);

  void change_directory(const std::string &path);

  const char* name()            const;
  const char* canonical_name()  const;
  const char* directory()       const;

  bool is_directory() const;
  bool is_file() const { return !is_directory(); }
};

}  // namespace aer

#endif  // AER_UTILS_PATH_H_
