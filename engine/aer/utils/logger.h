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
  ~Logger() {
    AER_DEBUG_CODE(display_stats();)
  }

  void message(const std::string &msg) {
    log(msg);
  }

  void warning(const std::string &msg) {
    if (log("[Warning]\n" + msg)) {
      warning_count_++;
    }
  }

  void error(const std::string &msg) {
    if (log("[Error]\n" + msg)) {
      error_count_++;
    }
  }

  void fatal(const std::string &msg) {
    error(msg);
    exit(EXIT_FAILURE);
  }

  void display_stats() {
    fprintf(stderr,
            "\n\x1b[2;31m" \
            "================= Logger stats =================\n" \
            "Warnings: %u\n" \
            "Errors: %u\n" \
            "================================================\n" \
            "\x1b[0m\n",
            warning_count_, error_count_);
  }


 private:
  bool log(const std::string &msg) {
    if (0u == error_log_.count(msg)) {
      fprintf(stderr, "\x1b[1;31m%s\x1b[0m\n", msg.c_str());
      error_log_[msg] = true;
      return true;
    }
    return false;
  }

  aer::U32 warning_count_ = 0u;
  aer::U32 error_count_   = 0u;
  std::unordered_map<std::string, bool> error_log_;

  friend class Singleton<Logger>;
};

}  // namespace aer

#endif  // AER_UTILS_LOGGER_H_
