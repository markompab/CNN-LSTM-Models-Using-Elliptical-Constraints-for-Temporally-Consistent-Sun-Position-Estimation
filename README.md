# CNN-LSTM-Models-Using-Elliptical-Constraints-for-Temporally-Consistent-Sun-Position-Estimation
A real-time sky imaging system and a CNN-LSTM network that regresses sun positions from input image sequences. In order to ensure that predicted sun positions are consistent with the natural path of the sun, the CNN-LSTM network was trained with elliptical constraints in addition to the primary L1-based loss term.

### CAU 2 Dataset
The fisheye images and their corresponding labels are available [here](https://drive.google.com/drive/folders/1SSUpzh1reuPbWOCHBRneHhZvN_g0c4IP?usp=sharing).
They were captured at Chung-Ang University in Seoul, South Korea. A 6.0 mm focal length and f3.2 aperture were used. The ISO was set to 200, and the ND Filter to 8. Shutter speeds of 1/30 and 1/60 were used. Ground truth labels (Elevation, Azimuth) were generated using Meeus' SPA algorithm.
