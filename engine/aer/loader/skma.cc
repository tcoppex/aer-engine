// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/loader/skma.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

#include "aer/common.h"
#include "aer/utils/path.h"



// Common -------------------------------------------------------------------

namespace {

void ReadHeader(FILE *fd, aer::ChunkHeader_t *pCH) {
  size_t nread = fread(pCH, sizeof(aer::ChunkHeader_t), 1, fd);
  AER_CHECK(nread == 1);

  AER_DEBUG_CODE(
  fprintf(stdout, "%s %u %u %u\n", pCH->id, pCH->flag,
                                   pCH->dataSize, pCH->dataCount);
  )
}

void ReadData(FILE *fd, const aer::ChunkHeader_t &ch, void **data) {
  if (ch.dataCount == 0u) {
    fprintf(stderr, "No data read for %s.\n", ch.id);
    return;
  }

  if (data == nullptr) {
    fseek(fd, ch.dataCount*ch.dataSize, SEEK_CUR);
    return;
  }

  *data = calloc(ch.dataCount, ch.dataSize);
  AER_CHECK(*data != nullptr);

  size_t nread = fread(*data, ch.dataSize, ch.dataCount, fd);
  AER_CHECK(nread == ch.dataCount);
}

aer::U32 ReadLine(FILE* fd, char buffer[], const aer::U32 N)
{
  aer::U32 i = 0u;
  while (!feof(fd) && (i+1<N) && ((buffer[i++] = fgetc(fd)) != '\n'));
  buffer[i] = '\0';
  return i;
}

bool FileExists(const std::string& filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}

}  // namespace



namespace aer {

// SKMA --------------------------------------------------------------------

SKMAFile::SKMAFile() :
  skmFile_(nullptr),
  skaFile_(nullptr),
  matFile_(nullptr)
{}

SKMAFile::~SKMAFile() {
  AER_SAFE_DELETE(skmFile_);
  AER_SAFE_DELETE(skaFile_);
  AER_SAFE_DELETE(matFile_);
}

bool SKMAFile::load(const char* filename) {
  bool bStatus = true;

  std::string basename(filename);

  // Load mesh data if any
  std::string skm_filename = basename + ".skm";
  if (FileExists(skm_filename)) {
    skmFile_ = new SKMFile();
    bStatus &= skmFile_->load(skm_filename.c_str());
  }

  // Load materials from file if any
  if (has_mesh() && skmFile_->numfacematerials() > 0) {
    std::string mat_filename = basename + ".mat";
    if (FileExists(mat_filename)) {
      matFile_ = new MATFile(/*skmFile_->numfacematerials()*/);
      bStatus &= matFile_->load(mat_filename.c_str());
    }
  }

  // Load animation data if any
  std::string ska_filename = basename + ".ska";
  if (FileExists(ska_filename)) {
    skaFile_ = new SKAFile();
    bStatus &= skaFile_->load(ska_filename.c_str());
  }

  return bStatus;
}


// SKM --------------------------------------------------------------------

SKMFile::SKMFile() :
  points_(nullptr),
  vertices_(nullptr),
  faces_(nullptr),
  vertex_materials_(nullptr),
  face_materials_(nullptr),
  bone_weights_(nullptr),
  skey_infos_(nullptr),
  skey_datas_(nullptr),
  ska_infos_(nullptr),

  numpoints_(0u),
  numvertices_(0u),
  numfaces_(0u),
  numvertexmaterials_(0u),
  numfacematerials_(0u),
  numboneweights_(0u),
  numskeyinfos_(0u),
  numskeydatas_(0u),
  numskainfos_(0u)
{}

SKMFile::~SKMFile() {
  AER_SAFE_FREE(points_);
  AER_SAFE_FREE(vertices_);
  AER_SAFE_FREE(faces_);
  AER_SAFE_FREE(vertex_materials_);
  AER_SAFE_FREE(face_materials_);
  AER_SAFE_FREE(bone_weights_);
  AER_SAFE_FREE(skey_infos_);
  AER_SAFE_FREE(skey_datas_);
  AER_SAFE_FREE(skey_infos_);
}

bool SKMFile::load(const char* filename) {
  char log_buffer[256];
  FILE *fd = fopen(filename, "rb");

  if (nullptr == fd) {
    sprintf(log_buffer, "\"%s\" does not exist", filename);
    AER_WARNING(log_buffer);
    return false;
  }

  // Retrieve the file size
  fseek(fd, 0L, SEEK_END);
  aer::U32 fileSize = ftell(fd);
  fseek(fd, 0L, SEEK_SET);

  // Check the file format
  ChunkHeader_t ch;
  ReadHeader(fd, &ch);

  if (strcmp(SKM_HEADERID_MAIN, ch.id) != 0) {
    sprintf(log_buffer, "\"%s\" is not a SKM file", filename);
    AER_WARNING(log_buffer);
    fclose(fd);
    return false;
  }

  // Load chunks of data
  while (!feof(fd) && (ftell(fd) < fileSize)) {
    // Read header
    ReadHeader(fd, &ch);

    // Load specific chunks
    if (strcmp(SKM_HEADERID_PNTS, ch.id) == 0) {
      numpoints_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&points_));
    } else if (strcmp(SKM_HEADERID_VERT, ch.id) == 0) {
      numvertices_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&vertices_));
    } else if (strcmp(SKM_HEADERID_FACE, ch.id) == 0) {
      numfaces_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&faces_));
    } else if (strcmp(SKM_HEADERID_VMAT, ch.id) == 0) {
      numvertexmaterials_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&vertex_materials_));
    } else if (strcmp(SKM_HEADERID_FMAT, ch.id) == 0) {
      numfacematerials_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&face_materials_));
    } else if (strcmp(SKM_HEADERID_BWGT, ch.id) == 0) {
      numboneweights_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&bone_weights_));
    } else if (strcmp(SKM_HEADERID_SKEYINFO, ch.id) == 0) {
      numskeyinfos_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&skey_infos_));
    } else if (strcmp(SKM_HEADERID_SKEYDATA, ch.id) == 0) {
      numskeydatas_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&skey_datas_));
    } else if (strcmp(SKM_HEADERID_SKAINFO, ch.id) == 0) {
      numskainfos_ = ch.dataCount; // should be no more than 1
      ReadData(fd, ch, reinterpret_cast<void**>(&ska_infos_));
    } else {
      sprintf(log_buffer, "Unknown chunk id \"%s\"", ch.id);
      AER_WARNING(log_buffer);

      // reads unknown data without storing them
      ReadData(fd, ch, nullptr);
    }
  }

  fclose(fd);

  return true;
}


// SKA --------------------------------------------------------------------

SKAFile::SKAFile() :
  bones_(nullptr),
  sequences_(nullptr),
  frames_(nullptr),

  numbones_(0u),
  numsequences_(0u),
  numframes_(0u)
{}

SKAFile::~SKAFile() {
  AER_SAFE_FREE(bones_);
  AER_SAFE_FREE(sequences_);
  AER_SAFE_FREE(frames_);
}

bool SKAFile::load(const char* filename) {
  char log_buffer[256];
  FILE *fd = fopen(filename, "rb");

  if (nullptr == fd) {
    sprintf(log_buffer, "\"%s\" does not exist", filename);
    AER_WARNING(log_buffer);
    return false;
  }

  // Retrieve the file size
  fseek(fd, 0L, SEEK_END);
  aer::U32 fileSize = ftell(fd);
  fseek(fd, 0L, SEEK_SET);

  // Check the file format
  ChunkHeader_t ch;
  ReadHeader(fd, &ch);

  if (strcmp(SKA_HEADERID_MAIN, ch.id) != 0) {
    sprintf(log_buffer, "\"%s\" is not a SKA file", filename);
    AER_WARNING(log_buffer);
    fclose(fd);
    return false;
  }

  // Load chunks of data
  while (!feof(fd) && (ftell(fd) < fileSize)) {
    // Read header
    ReadHeader(fd, &ch);

    // Load specific chunks
    if (strcmp(SKA_HEADERID_BONE, ch.id) == 0) {
      numbones_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&bones_));
    } else if (strcmp(SKA_HEADERID_SEQU, ch.id) == 0) {
      numsequences_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&sequences_));
    } else if (strcmp(SKA_HEADERID_FRAM, ch.id) == 0) {
      numframes_ = ch.dataCount;
      ReadData(fd, ch, reinterpret_cast<void**>(&frames_));
    } else {
      sprintf(log_buffer, "unknown chunk id \"%s\".\n", ch.id);
      AER_WARNING(log_buffer);

      // reads unknown data without storing them
      ReadData(fd, ch, nullptr);
    }
  }

  fclose(fd);

  return true;
}


// MAT --------------------------------------------------------------------

bool MATFile::load(const char* filename) {
  filepath_ = aer::Path(filename).directory();
  
  FILE *fd = nullptr;
  if (!(fd = fopen(filename, "r"))) {
    fprintf( stderr, "Error: \"%s\" cannot be found.\n", filename);
    return false;
  }

  material_datas_.resize(64u); // XXX bad baaad bad bad bad
  const char *TexturePrefix[] = {"Kd", "Ks", "Bump"};

# define MAT_MAXLINESIZE 256u
  char line[MAT_MAXLINESIZE];

  aer::U32 i = 0u;
  while (!feof(fd)) {
    MaterialData &material_data = material_datas_[i];

    fscanf(fd, "%s\n", material_data.name);

    while(ReadLine(fd, line, MAT_MAXLINESIZE) > 1u) {
      char type[MAT_TYPENAME_SIZE];
      char path[MAT_TEXTUREPATH_SIZE];
      sscanf(line, "%s %s\n", type, path);

      for (aer::U32 p = 0u; p < kNumMaterialType; ++p) {
        if (!strcmp(type, TexturePrefix[p])) { 
          material_data.mList[p] = new char[MAT_TEXTUREPATH_SIZE];
          strcpy(material_data.mList[p], path);
        }
      }
    }
    ++i;
  }
# undef MAT_MAXLINESIZE

  material_datas_.resize(i); // xxx

  fclose(fd);

  return true;
}

const char* MATFile::material_from_name(const std::string& name,
                                        const MaterialType type) const {
  for (aer::U32 i(0u); i < material_datas_.size(); ++i) {
    if (!strcmp(material_datas_[i].name, name.c_str())) {
      return material_datas_[i].mList[type];
    }
  }
  return nullptr;
}

}  // namespace aer
