-- original RGB image (3 channels) with pixel values in [0, 255]
type UnnormalisedImage = Tensor Real [28, 28, 3]

-- image normalised with mean / std normalisation (common when training computer vision networks)
type NormalisedImage = Tensor Real [28, 28, 3]

-- normalisation:
mean : Vector Real 3
mean = [0.7382130772013564, 0.7234146030672689, 0.6367749837159091]

std : Vector Real 3
std = [0.3207768071065142, 0.2532202588391356, 0.28771851114769775]

-- normalisation: x_norm = (x - mean) / std
normalise : UnnormalisedImage -> NormalisedImage
normalise x =
  foreach i .
    foreach j .
      foreach k .
        let m = mean ! k in
        let s = std ! k in
          ( (x ! i ! j ! k) - m ) / s

-- denormalisation: x = x_norm * std + mean
denormalise : NormalisedImage -> UnnormalisedImage
denormalise x =
  foreach i .
    foreach j .
      foreach k .
        let m = mean ! k in
        let s = std ! k in
          ( (x ! i ! j ! k) * s ) + m

-- pixel values between 0 and 255
validImage : UnnormalisedImage -> Bool
validImage x = forall i j k . 0 <= x ! i ! j ! k <= 255

@network
classifier : NormalisedImage -> Tensor Real [6]

-- a face is predicted if the logit is >= 0
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

-- not both opposite faces (e.g. 1 and 6) can be visible at the same time
-- but since all images show 3 faces of the dice, exactly one of each opposing face pairs needs to be visible
oneOrOther : UnnormalisedImage -> Index 6 -> Index 6 -> Bool
oneOrOther image a b = (predicts image a) or (predicts image b)

predictsExactlyOneOf : UnnormalisedImage -> Index 6 -> Index 6 -> Bool
predictsExactlyOneOf image face1 face2 =
  ((predicts image face1) and not (predicts image face2)) or
  ((predicts image face2) and not (predicts image face1))

oppositeFacePairs : Vector (Vector (Index 6) 2) 3
oppositeFacePairs = [ [0, 5], [1, 4], [2, 3] ]

oppositeFaces : UnnormalisedImage -> Bool
oppositeFaces image = forall perturbation .
  let perturbedImage = image + perturbation in
    boundedByEpsilon perturbation and validImage perturbedImage =>
      (forall p .
        let pair = oppositeFacePairs ! p in
          predictsExactlyOneOf perturbedImage (pair ! 0) (pair ! 1) )

@property
robust : Vector Bool n
robust = foreach i . oppositeFaces (images ! i)
