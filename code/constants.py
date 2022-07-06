

from scipy.stats import norm

confidence = 0.95
quantile = norm.ppf((1 + confidence)/2)
