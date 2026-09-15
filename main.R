library(here)

path <- "data/raw"
files <- list.files(path, pattern="\\.csv", full.names=TRUE)

# data <- vector("list", length(files))

# for (i in seq_along(files)){
  # print(paste(path, "/", file, sep=""))
  # file <- files[i]
  # data <- read.csv(paste(path, "/", file, sep=""))
# }

# typeof(data)

# best way to load data
data <- lapply(files, read.csv)
names(data) <- sub("\\,csv$", "", basename(files))

data$Danio_rerio

