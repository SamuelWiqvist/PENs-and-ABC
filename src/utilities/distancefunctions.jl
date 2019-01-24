
function euclidean_dist(s_star::Vector, s::Vector, w::Vector)

  Δs =  (s_star-s)
  dist = 0

  for i in 1:length(Δs)
    dist = dist + 1/w[i]^2*Δs[i]^2
  end

  return sqrt(dist)

end
