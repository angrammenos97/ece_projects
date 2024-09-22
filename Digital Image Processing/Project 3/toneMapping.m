function tonedImage = toneMapping(radianceMap , gamma)
  minRad = min(radianceMap, [], [1 2]);
  maxRad = max(radianceMap, [], [1 2]);
  radianceMapNorm = (radianceMap-minRad)./(maxRad-minRad);
  tonedImage = uint8((radianceMapNorm.^gamma)*255.0);
end