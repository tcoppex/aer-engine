// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_UTILS_LOGGER_H_
#define AER_UTILS_LOGGER_H_

#include <cstdio>
#include <string>
#include <unordered_map>

#include "aer/common.h"
#include "aer/utils/singleton.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// A basic logger.
/// Could be use inside loops to print messages only once
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Logger : public Singleton<Logger> {
 public:
  void warning(const std::string &msg) {
    log("Warning : " + msg);
  }

  void error(const std::string &msg) {
    log("Fatal Error : " + msg);
    exit(EXIT_FAILURE);
  }

 private:
  void log(const std::string &msg) {
    if (0u == error_log_.count(msg)) {
      fprintf(stderr, "\x1b[1;31m%s\x1b[0m.\n", msg.c_str());
    }
    error_log_[msg] = true;
  }

  std::unordered_map<std::string, bool> error_log_;

  friend class Singleton<Logger>;
};

}  // namespace aer

#endif  // AER_UTILS_LOGGER_H_
