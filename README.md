# Convection_trigger_function

## MOTIVATION
- CAM7/ZM scheme relies on CAPE-based trigger prone to convection overprediction
- Tropical Overshooting Clouds Analysis (TOOCAN) dataset provides tracked mesoscale convective systems from geostationary obs and GPM (Precipitation measurement) satellite

### APPROACH
- Train an XGBoost, neural network or other ML classifiers to predict convection onset from large-scale env. variables
- Evaluate offline against TOOCAN labels 

