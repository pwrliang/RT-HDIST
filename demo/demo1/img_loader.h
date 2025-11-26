#pragma once

#include <string>
#include <vector>

#include "itkDirectedHausdorffDistanceImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"


std::vector<float3> LoadImage(
    const std::string& path, itk::Size<3>& size,
    int limit = std::numeric_limits<int>::max()) {
  using PixelType = unsigned char;
  using ImageType = itk::Image<PixelType, 3>;
  using ReaderType = itk::ImageFileReader<ImageType>;
  using point_t = float3;
  std::vector<point_t> points;

  auto reader = ReaderType::New();
  reader->SetFileName(path);
  try {
    reader->Update();
  } catch (itk::ExceptionObject& err) {
    std::cerr << "Error: " << err << std::endl;
    return {};
  }
  auto image = reader->GetOutput();
  // Iterator over the image
  using IteratorType = itk::ImageRegionIterator<ImageType>;
  size = image->GetLargestPossibleRegion().GetSize();

  IteratorType it(image, image->GetLargestPossibleRegion());

  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
    PixelType value = it.Get();
    if (value > 0) {  // Non-empty voxel
      auto index = it.GetIndex();
      point_t p;

      for (int dim = 0; dim < 3; ++dim) {
        (&p.x)[dim] = index[dim];
      }
      points.push_back(p);
      if (points.size() >= limit) {
        break;
      }
    }
  }
  return points;
}

