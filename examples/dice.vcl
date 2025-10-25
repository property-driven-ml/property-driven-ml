-- original RGB image (3 channels) with pixel values in [0, 255]
type UnnormalisedImage = Tensor Real [3, 28, 28]

-- image normalised with mean / std normalisation (common when training computer vision networks)
type NormalisedImage = Tensor Real [3, 28, 28]

-- normalisation:
mean : Vector Real 3
mean = [0.7232479453086853, 0.7257601618766785, 0.6415771842002869]

std : Vector Real 3
std = [0.32942402362823486, 0.24738003313541412, 0.2831753194332123]


-- normalisation: x_norm = (x - mean) / std
normalise : UnnormalisedImage -> NormalisedImage
normalise x =
  foreach c .
    foreach h .
      foreach w .
        let m = mean ! c in
        let s = std ! c in
          ( (x ! c ! h ! w) - m ) / s

-- denormalisation: x = x_norm * std + mean
denormalise : NormalisedImage -> UnnormalisedImage
denormalise x =
  foreach c .
    foreach h .
      foreach w .
        let m = mean ! c in
        let s = std ! c in
          ( (x ! c ! h ! w) * s ) + m

-- pixel values between 0 and 255
validImage : UnnormalisedImage -> Bool
validImage x = forall c h w . 0 <= x ! c ! h ! w <= 255

@network
classifier : NormalisedImage -> Tensor Real [6]

-- a label i is predicted if its logit y_i > 0
predicts : UnnormalisedImage -> Index 6 -> Bool
predicts x i = ( classifier (normalise x) ) ! i > 0

@parameter
epsilon : Real

boundedByEpsilon : UnnormalisedImage -> Bool
boundedByEpsilon x = forall i j k . -epsilon <= x ! i ! j ! k <= epsilon

@parameter(infer=True)
n : Nat

@dataset
images : Vector UnnormalisedImage n

oppositeFacePairs : Vector (Vector (Index 6) 2) 3
oppositeFacePairs = [ [0, 5], [1, 4], [2, 3] ]

oppositeFaces : UnnormalisedImage -> Bool
oppositeFaces image = forall perturbation .
  let perturbedImage = image + perturbation in
    boundedByEpsilon perturbation and validImage perturbedImage =>
      (forall p .
        let pair = oppositeFacePairs ! p in
          not ( (predicts perturbedImage (pair ! 0) ) and (predicts perturbedImage (pair ! 1) ) )
      )

@property
robust : Vector Bool n
robust = foreach i . oppositeFaces (images ! i)
