// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include <sys/stat.h>
#include <string>

#include "aer/loader/texture_2d_proxy.h"
#include "aer/loader/image_2d.h"
#include "aer/utils/logger.h"


namespace {

bool FileExists(const std::string& filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}

}  // namespace


namespace aer {

Texture2D* Texture2DProxy::load(const std::string& id) {
  Image2D img;

  if (!FileExists(id) || !img.load(id.c_str())) {    
    Logger::Get().warning("Texture \"" + id + "\" not found");
    return nullptr;
  }

  Texture2D *pTexture = new Texture2D();
  pTexture->generate();
  pTexture->bind();
  pTexture->allocate(img.internalformat(), img.width(), img.height());
  pTexture->upload(img.format(), img.type(), img.data());
  aer::Texture::Unbind(GL_TEXTURE_2D);

  return pTexture;
}

}  //  namespace aer