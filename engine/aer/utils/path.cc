// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/utils/path.h"

#ifdef AER_WINDOWS
# include <direct.h>
# define AER_GETPWD()   _getcwd( NULL, 0)
#else
# include <unistd.h>
# define AER_GETPWD()   getcwd( NULL, 0)
#endif

#define DIR_SEPARATOR   '/'


namespace aer {

void Path::CleanFilename(std::string &filename) {
  size_t pos;

  // replaces Windows' antislash by Unix' slash
  while ((pos=filename.find_first_of('\\')) != std::string::npos) {
    filename.replace( pos, 1u, 1u, DIR_SEPARATOR );
  }

  // clean previous dir marker
  while ((pos=filename.find("..")) != std::string::npos) {
    if (pos <= 1) {
      filename.replace(0, pos+3, 1, DIR_SEPARATOR);
      continue;
    }

    size_t prev_dir = filename.find_last_of(DIR_SEPARATOR, pos-2) + 1;
    filename.erase(prev_dir, (pos-prev_dir) + 3);
  } 

  // clean current dir marker
  while ((pos=filename.find("./")) != std::string::npos) {    
    filename.erase(pos, 2);
  }
}

Path::Path(const std::string &pathname) : 
  name_idx_(0u) 
{
  char *cwd = AER_GETPWD();
  canonical_ = cwd;
  canonical_ += DIR_SEPARATOR;
  change_directory(pathname);  
  AER_SAFE_FREE(cwd);
}

Path::Path(const Path &path) : 
  canonical_(path.canonical_),
  name_idx_(path.name_idx_)
{}

void Path::change_directory(const std::string &path) {
  if (path.size() == 0u) {
    return;
  }

  if (path[0] == DIR_SEPARATOR) {
    canonical_ = std::string(path);
  } else {
    canonical_ += path;
  }

  CleanFilename(canonical_);   

  // Name index
  if (is_directory()) {
    name_idx_ = canonical_.find_last_of(DIR_SEPARATOR, canonical_.length()-2) + 1;  
  } else {
    name_idx_ = canonical_.find_last_of(DIR_SEPARATOR) + 1;  
  }

  canonical_directory_ = canonical_.substr(0, name_idx_);//
}

const char* Path::name() const {
  return &(canonical_.c_str()[name_idx_]);
}

const char* Path::directory() const {
  return canonical_directory_.c_str();
}

const char* Path::canonical_name() const {
  return canonical_.c_str();
}

bool Path::is_directory() const {
  return canonical_.at(canonical_.length()-1u) == DIR_SEPARATOR;
}

}  // namespace aer
