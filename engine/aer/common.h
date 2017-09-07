// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_COMMON_H_
#define AER_COMMON_H_

/*
Il échappa ainsi, 
parce que son coeur était pur, 
au mal qui se tramait contre lui.
*/

/// includes headers use by the entire library
#include "aer/core/defs.h"
#include "aer/core/types.h"
#include "aer/core/algebra_types.h"

/// Bypass the simple warning macro to use the engine logger
#if AER_DEBUG
# undef AER_WARNING
# include "aer/utils/logger.h"
# define AER_WARNING(msg)     aer::Logger::Get().warning(msg);
#endif

#endif  // AER_COMMON_H_
