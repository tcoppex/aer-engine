// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_MATERIAL_PROXY_H
#define AER_LOADER_MATERIAL_PROXY_H

#include <string>

#include "aer/aer.h"
#include "aer/memory/resource_proxy.h"
#include "aer/rendering/material.h"

#include "aura/skma_loader/skma_loader.h"


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Manage material resources
/// (don't seems to work very well)
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class MaterialProxy : public aer::ResourceProxy<aer::Material> {
 public:
  // load("materialfile.objname") load every materialfile.*
  // if objname is not found, and return objname
  virtual aer::Material* load(const std::string& id) override {
    size_t separator_idx = id.find_last_of('.');
    std::string name(id.substr(0u, separator_idx));

    if (load_file(name)) {
      return files_[id];
    }

    return nullptr;
  }


 private:
  ///
  bool load_file(const std::string &name) {
    MATFile matFile;

    std::string filename = name + ".mat";
    if (!matFile.load(filename.c_str())) {
      return false;
    }

    setup_materials(name, matFile);
    return true;
  }

  ///
  void setup_materials(const std::string &name,
                       const MATFile &matFile)
  {
    std::string texture_directory(matFile.filepath());
    texture_directory += "textures/"; //

    char reference_id[256];
    int ref_size = sprintf(reference_id, "%s.", name.c_str());
    char *reference_id_end = &reference_id[ref_size];

    const char *texture_name = nullptr;

    for (aer::U32 i = 0u; i < matFile.count(); ++i) {      
      aer::Material *material = new aer::Material();      

      texture_name = matFile.material_from_id(i, MATFile::TEXTURE_DIFFUSE);
      if (texture_name != nullptr) {
        std::string texture_path(texture_directory + texture_name);
        material->set_diffuse_map_id(texture_path);
      }

      //
      sprintf(reference_id_end, "%s", matFile.material_name(i));
      files_[reference_id] = material;
      reference_counts_[material] = 0u;
    }
  }
};


#endif  // AER_LOADER_MATERIAL_PROXY_H
